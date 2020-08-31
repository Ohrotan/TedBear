#-*- coding: utf-8 -*-
import hydra
#커맨드라인 처리,Config 관리,원격실행,로깅 등의 귀찮은 부분을 대신함.
from hydra import utils
from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import os


#os.chdir('zerospeech')
print("pre",os.getcwd())
def preemphasis(x, preemph):# 고주파에서의 잡음을 줄이기 위한 전처리라고 합니다
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)
# VAE에서 디코더 쪽에 들어가는거같습니다


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x
# VAE에서 디코더 쪽에서 나오는거같습니다

def process_wav(wav_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=sr,
                          offset=offset, duration=duration)#suffix 마지막 구성 요소의 파일 확장자, witg_suffix는 그 확장자로 지정.
    # 따라서 wav파일을 로드하는 것. wav에  노말라이즈된 오디오신호 저장, _에 sampling rate 저장
    # offset은 오디오를 몇 초 이후 부터 로드할건지. 
    # duration은 오디오를 몇초까지 로드할건지
    
    
    wav = wav / np.abs(wav).max() * 0.999
    # 최대값으로 나눠 원소들간의 변동을 명확화시킨다 

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph), # 고주파 잡음 줄인거로 시행
                                         sr=sr,
                                         n_fft=n_fft, # 푸리에 변환을 한번 당 시행 할 윈도우의 크기. 변환할 때의 크기지 사용할 떄의 크기가 아님
                                         n_mels=n_mels, 
                                         #n_mels가 곧 칼라맵의 주파수 해상도가 됩니다. 
                                         #즉, mel 기준으로 몇개의 값으로 표현할지를 나타내는 변수입니다.
                                         #많으면 많을수록 원래 stft로 표현한 칼라맵과 유사한 형태가 됩니다. 
                                         #그만큼 필터를 촘촘하게 쓰는 것 같습니다.
                                         hop_length=hop_length, # 스트라이드
                                         win_length=win_length, # 윈도우의 크기. 디폴트는 n_fft값. 
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db) # 데시벨스케일드 melspectrogram으로 변환. 로그 변환
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    return out_path, logmel.shape[-1]

def userid(name):
    global user_id
    user_id = name

@hydra.main(config_path="zerospeech/config/preprocessing.yaml")
# 기본 경로를 야믈형식으로 저장해뒀다. 이걸로 바로 해당 파일로 접근. 여기서 in_dir이 ??? 로 되어있음
 # 위의 confg_path의 파일의 모든 값이 함수의 인자로 들어오는 것
def preprocess_dataset(cfg):
    print("in_dir:",cfg.in_dir)
    in_dir = Path(utils.to_absolute_path(cfg.in_dir)) # Path로 경로를 지정(객체로 지정).


    #utils.to_absolute_path는 경로를 절대 경로(프로그램이 실행되는 최상단, 여기선 zerospeech폴더부터 표현하는 것)로 변경.
    # config/dataset/2019/english 안에있는 in_dir변수를 경로로 만든다.
    out_dir = Path(utils.to_absolute_path("datasets")) 
    # cfg.dataset.dataset: 2019/english을 보아 datasets/2019/english에 out_dir 지정해야하나
    out_dir.mkdir(parents=True, exist_ok=True)
    #  Create a new directory at this given path.
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    # 병렬작업 실행자
    for split in ["train"]:
        print("Extracting features for {} set".format(split)) 
        futures = []
        split_path = out_dir / cfg.dataset.language / split # 위에서 정해진 out_dir 하위에 train과 test가 차례로 split_path가 됨, cfg.dataset.language는 config/preprocessing.yaml 안에 dataset: 2019/english라고 되어있는데 이 부분은 config/dataset/2019/english.yaml 파일로 연결됨, 이안에 language 변수 있음
        with open(split_path.with_suffix(".json")) as file: # train.json, test.json 차례로 불러오게됨
            metadata = json.load(file)
            for in_path, start, duration, out_path in metadata: # json 파일 열어보면 순서대로 in_path, start, duration, out_path 정보를 주고 있음, 그러니 우리가 커스터마이징할 때 이 json 파일도 변경해야함
                wav_path = in_dir / in_path
                print("wav_path:",wav_path)
                out_path = out_dir / out_path
                print("out_path:",out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                futures.append(executor.submit(
                    partial(process_wav, wav_path, out_path, **cfg.preprocessing,
                            offset=start, duration=duration))) # partieal로 위에 정의한 함수들 실행

        results = [future.result() for future in tqdm(futures)]
    

        lengths = [x[-1] for x in results]
        frames = sum(lengths)
        frame_shift_ms = cfg.preprocessing.hop_length / cfg.preprocessing.sr #160,16000
        hours = frames * frame_shift_ms / 3600
        print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


if __name__ == "__main__":
    preprocess_dataset()
