import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm

import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeechDataset
from model import Encoder, Decoder

# training 중간과정을 기록함, 중단되어도 다시 그 지점부터 시작할 수 있도록
def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
# 이 중간과정을 저장할 폴더 경로와 파일명
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
# 객체를 파일로 저장하는 함수
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


@hydra.main(config_path="config/train.yaml")
def train_model(cfg): # 위의 confg_path의 파일의 모든 값이 cfg로 함수의 인자로 들어오는 것
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir)) # chekpoint dir 지정
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda 사용이 가능하지 않으면 cpu 사용

# 객체 생성시 초기값을 config/model/default.yaml 에서 가져옴
    encoder = Encoder(**cfg.model.encoder) 
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

# 최적화할 때 Adam이라는 알고리즘을 사용
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# ref: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=cfg.training.optimizer.lr)

#amp는 tensor core의 트레이닝을 가속화시켜주는 도구
    [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
# 학습률을 최적화하는 스케줄러
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)


 # training을 중간에 멈췄을 경우 이 값을 True로 변경하면 checkpoint에서 기록을 가져와서 다시 시작. 처음 training하는 경우에는 False로 값이 설정되어 있음
    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path 
# config/train.yaml에서 dataset: 2019/english 로 되어있으므로 config/dataset/2019/english.yaml 에서 path 참조
    dataset = SpeechDataset(
        root=root_path,
	# preprocessing: default 라고 되어있으므로 config/preprocessing/default.yaml에서 hop_length, sr, smaple_frames 참조
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr,
        sample_frames=cfg.training.sample_frames)

# training: default 라고 되어있으므로 config/training/default.yaml에서 batch_size,n_workers,n_steps 참조
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)

    n_epochs = cfg.training.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            z, vq_loss, perplexity = encoder(mels)
            output = decoder(audio[:, :-1], z, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

# cfg.training.checkpoint_interval 에서 지정한 횟수마다 주기적으로 트레이닝 중간 과정 checkpoints에 기록하여서 갑자기 중단되더라도 복구할 수 있게 백업본 만들기
            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, amp,
                    scheduler, global_step, checkpoint_dir)
# training 결과 기록
        writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
        writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))


if __name__ == "__main__":
    train_model()
