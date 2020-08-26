import hydra
import hydra.utils as utils

import json
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm
import pyloudnorm # 볼륨 평준화 라이브러리? 볼륨 조절할 때 쓰는 듯

from preprocess import preemphasis
from model import Encoder, Decoder


@hydra.main(config_path="config/convert.yaml")
def convert(cfg):
    dataset_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path    #zerospeech/datasets/2019/english
    with open(dataset_path / "speakers.json") as file: # 말하는 사람들 이름 써있는 데이터
        speakers = sorted(json.load(file)) # speakers라는 객체로 저장

    synthesis_list_path = Path(utils.to_absolute_path(cfg.synthesis_list)) # ???인걸 보니 우리가 파이썬에서 돌릴때 지정해줘야함
    with open(synthesis_list_path) as file:
        synthesis_list = json.load(file) # datasets/2019/english에 있는 synthesis.json보면됨

    in_dir = Path(utils.to_absolute_path(cfg.in_dir)) # ???임. zerospeech 폴더로 경로따면 될듯. (./)
    out_dir = Path(utils.to_absolute_path(cfg.out_dir)) #???임. 목소리 변환된 결과를 저장할 경로
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu안되면 cpu로

    encoder = Encoder(**cfg.model.encoder) #ZeroSpeech/config/model/default에 있는 encoder
    decoder = Decoder(**cfg.model.decoder) #ZeroSpeech/config/model/default에 있는 decoder
    encoder.to(device) # cpu or gpu
    decoder.to(device) # cpu or gpu

    print("Load checkpoint from: {}:".format(cfg.checkpoint)) ### ???로 되어있는데 pretrained, 혹은 checkpoint까지 학습된 모델 있으면 그 모델의 위치로 지정
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # checkpoint에 지정된 weight들을 불러옵니다
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    meter = pyloudnorm.Meter(cfg.preprocessing.sr) #sr:16000으로 조정??  https://www.christiansteinmetz.com/projects-blog/pyloudnorm 소음 관련같습니다..

    for wav_path, speaker_id, out_filename in tqdm(synthesis_list):  #"english/test/S002_0379088085","V002","V002_0379088085"
        wav_path = in_dir / wav_path # ./english/test/S002_0379088085
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),
            sr=cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(wav) #인풋의 음량을 측정인듯
        wav = wav / np.abs(wav).max() * 0.999 

        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1

        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)#unsqueeze()함수는 인수로 받은 위치에 새로운 차원을 삽입
        
        #https://subinium.github.io/pytorch-Tensor-Variable/#%EB%8D%94%EB%AF%B8-%EC%B0%A8%EC%9B%90-%EC%B6%94%EA%B0%80%EC%99%80-%EC%82%AD%EC%A0%9C--squeeze--unsqueeze
        
        #https://datascienceschool.net/view-notebook/4f3606fd839f4320a4120a56eec1e228/
        
        
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device) # 마찬가지로 텐서로 만드는데
        
        #텐서에는 자료형이라는 것이 있습니다. 각 데이터형별로 정의되어져 있는데,
        #예를 들어 32비트의 유동 소수점은 torch.FloatTensor를, 64비트의 부호 있는 정수는 torch.LongTensor를 사용합니다.
        #GPU 연산을 위한 자료형도 있습니다. 예를 들어 torch.cuda.FloatTensor가 그 예입니다.
        
        # 즉 mel은 소수점있고 speaker는 소숫점 없으니까!
        with torch.no_grad(): # 자동미분,벡터연산한 결과의 연산기록 추적못하게 https://bob3rdnewbie.tistory.com/315
            z, _ = encoder.encode(mel)
            output = decoder.generate(z, speaker)

        output_loudness = meter.integrated_loudness(output) #아웃풋의 음량을 측정인듯
        output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
        # 아웃풋의 음량을 input에 넣은 wav의 음량과 동일하게 변경
        path = out_dir / out_filename
        librosa.output.write_wav(path.with_suffix(".wav"), output.astype(np.float32), sr=cfg.preprocessing.sr)


if __name__ == "__main__":
    convert()
