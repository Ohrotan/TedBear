import numpy as np
import torch
from torch.utils.data import Dataset
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sr, sample_frames):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames

# root path 아래 있는 speaker.json 파일 열어서 self.speaker에 저장
        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

# 최소 시간
        min_duration = (sample_frames + 2) * hop_length / sr
        with open(self.root / "train.json") as file:
            metadata = json.load(file)

# 최소 duration 보다 큰 경우에 해당되는 out_path 값을 train.json에서 읽어옴
            self.metadata = [
                Path(out_path) for _, _, duration, out_path in metadata
                if duration > min_duration
            ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
# index에 해당되는 out_path 찾아옴
        path = self.metadata[index]
        path = self.root.parent / path

# 그 path를 이용하여 wav, mel 파일 읽기
        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

# pos에 1 부터 ~ 사이의 랜덤값
# 랜덤인 pos 값을 이용하여 mel과 audio에서 부분적으로 데이터 추출
# 왜인지는 모르겠음..
        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = self.speakers.index(path.parts[-2])

# torch 에서 쓰는 타입으로 변경하여 리턴
        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker
