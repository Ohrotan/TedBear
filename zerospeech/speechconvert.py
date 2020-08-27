#!/usr/bin/env python
# coding: utf-8

import os
print("speech convert:",os.getcwd())
#import preprocess
import convert
import hydra
import editconfig
def get_converted_audio(user_id, user_audio_path, org_audio_path,start_transcript,end_transcrpit) : #아래 함수들을 한번에 실행
    editconfig.speaker_json(user_audio_path, org_audio_path)
    editconfig.train_json(user_audio_path)
    editconfig.test_json(org_audio_path)
    editconfig.synthesis_json(user_id, org_audio_path,start_transcript,end_transcrpit)
    convert.convert()
    hydra._internal.hydra.GlobalHydra().clear()
