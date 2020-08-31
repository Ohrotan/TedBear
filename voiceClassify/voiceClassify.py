import os
from os import walk
import shutil
import numpy as np
from voiceClassify.utils import *
from pickle import load,dump
from sklearn.preprocessing import normalize


class VoiceClassify():

    def __init__(self,user_audio_file):
        
        with open("../voiceClassify/model/clf-xgboost.pkl","rb") as f: # xgboost 모델 불러오기
            self.clf=load(f)
        self.user_audio_file=user_audio_file
        
    @staticmethod
    def _process(self): # 특징추출

        try:
 
            result =[]
            mfccs,  mel,  tonnetz,_ = extract_features(self.user_audio_file) # 라벨은 없으므로 _ 처리
            re = np.hstack((normalize(mfccs),normalize(mel),normalize(tonnetz)))
            result.append(re)
 
            return result

        except Exception as e:

            print(e)
            
    def classes_(self): # 102개의 class존재

        return self.clf.classes_

    def predict(self): # 예측하기

        dps = self._process(self)
        
        result = []
        for dp in dps:
            prediction,prob,classes = self.clf.predict(dp),self.clf.predict_proba(dp),self.clf.classes_
            result.append((prediction,prob,classes)) 
        
        #리스트형식으로 예측된 class, 각 class들의 확률, 각 class 이름 반환.
        
        return result 



    
