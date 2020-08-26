#!/usr/bin/env python
# coding: utf-8


import json

from pathlib import Path

import json

import wave
import contextlib
import os
import re
import math


# Speaker.json 바꾸기
def speaker_json(user_audio_path, org_audio_path):
    # 일단 train폴더 안의 speaker를 추가(웹 이용자가 추가될 것)
    path_dir = str(user_audio_path) #경로 끝에 / 꼭 붙이기 

    file_list =os.listdir(path_dir) #경로 읽어 파일명 리스트 만들기
    file_list.sort() #정렬
    with open("./datasets/english/speakers.json", "r") as st_json: #json 파일 읽기
        speakers = json.load(st_json)
        for i in file_list:
            a=i.index('_') #file_list 형식이 이름_번호 형식이라 잘라야함
            speaker=i[:a]
            if speaker not in speakers: #speakers.json에 없으면 추가
                speakers.append(speaker)
    with open("./datasets/english/speakers.json", 'w', encoding='utf-8') as make_file:

        json.dump(speakers, make_file, indent="\t") # 추가성공

        # test 폴더안의 speaker 추가(ted 영상 speecher)
    path_dir =str(org_audio_path)#경로 끝에 / 꼭 붙이기

    file_list =os.listdir(path_dir) #경로 읽어 파일명 리스트 만들기
    file_list.sort() #정렬
    with open("./datasets/english/speakers.json", "r") as st_json:#json 파일 읽기
        speakers = json.load(st_json)
        for i in file_list:
            a=i.index('_')#file_list 형식이 이름_번호 형식이라 잘라야함
            speaker=i[:a]
            if speaker not in speakers:#speakers.json에 없으면 추가
                speakers.append(speaker)
    with open("./datasets/english/speakers.json", 'w', encoding='utf-8') as make_file:

        json.dump(speakers, make_file, indent="\t") #추가성공

    

def train_json(user_audio_path): # 유저 정보를 기록하는 train.json 수정하기
    path_dir = str(user_audio_path)#경로 끝에 / 꼭 붙이기
    file_list =os.listdir(path_dir) #경로 읽어 파일명 리스트 만들기
    file_list.sort() #정렬

    with open("./datasets/english/train.json", "r") as st_json:
        train = json.load(st_json)
        for i in file_list:
            with contextlib.closing(wave.open(path_dir + i,'r')) as f: # wav파일 읽기
                frames = f.getnframes()  #오디오 프레임의 수를 반환
                rate = f.getframerate()  #샘플링 빈도를 반환
                duration = frames / float(rate) #프레임/샘플링빈도 = 오디오의 길이
                duration=math.floor(duration*100) 
                duration=duration/100 #소숫점 처리 완료
                fileplace=path_dir[2:]+i[0:-4] #파일의 위치
                filepreprocessplace=i.index('_')
                a=[str(fileplace),0.0,duration,path_dir[2:16]+i[:filepreprocessplace]+'/'+i[0:-4]]
                #a=[파일의 위치, 오디오시작시간(0.0으로 통일), 오디오 길이, 전처리 시 저장될 위치]
                if a not in train:
                    train.append(a) # json에 없으면 추가

    with open("./datasets/english/train.json", 'w', encoding='utf-8') as make_file:

        json.dump(train, make_file, indent="\t") #json에 추가 완료


def test_json(org_audio_path): # ted오디오의 정보를 기록하는 test.json 수정
    path_dir =  str(org_audio_path)#경로 끝에 / 꼭 붙이기
    file_list =os.listdir(path_dir) #경로 읽어 파일명 리스트 만들기
    file_list.sort()
    with open("./datasets/english/test.json", "r") as st_json:
        test = json.load(st_json)
        for i in file_list:
            with contextlib.closing(wave.open(path_dir + i,'r')) as f: # wav파일 읽기
                frames = f.getnframes()#오디오 프레임의 수를 반환
                rate = f.getframerate() #샘플링 빈도를 반환
                duration = frames / float(rate)#프레임/샘플링빈도 = 오디오의 길이
                duration=math.floor(duration*100)
                duration=duration/100 #소숫점 처리 완료
                fileplace=path_dir[2:]+i[0:-4]#파일의 위치
                filepreprocessplace=i.index('_')
                a=[str(fileplace),0.0,duration,path_dir[2:16]+i[:filepreprocessplace]+'/'+i[0:-4]]
                #a=[파일의 위치, 오디오시작시간(0.0으로 통일), 오디오 길이, 전처리 시 저장될 위치]
                if a not in test:
                    test.append(a)

    with open("./datasets/english/test.json", 'w', encoding='utf-8') as make_file:

        json.dump(test, make_file, indent="\t")  #json에 추가 완료


def synthesis_json(user_id, org_audio_path): # 음성 합성할 때 쓸 synthesis_list.json 변경
    user_name=str(user_id) #user_id로 speaker이용
    path_dir=str(org_audio_path) #변환 대상이 될 ted영상의 경로
    file_list=os.listdir(path_dir) 
    with open("./datasets/english/synthesis_list.json", "r") as st_json: 
        synthesis = json.load(st_json)
        synthesis=[] #초기화
        for i in file_list:
            fileplace=path_dir[2:]+i[0:-4] #ted영상의 path
            plus=re.split(r'/', i) 
            plus=plus[-1][:-4] #필요한 정보만 슬라이스. 
            save_name=user_name+'_'+plus # 변환된 파일 이름
            a=[str(fileplace),user_name,save_name] 
            if a not in synthesis:
                    synthesis.append(a)

    with open("./datasets/english/synthesis_list.json", 'w', encoding='utf-8') as make_file:

        json.dump(synthesis, make_file, indent="\t") # 추가 완료

