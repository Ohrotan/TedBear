# 영어 스피킹을 위한 

# Self-shadowing 지도 학습 프로그램



팀: TedBear ( 강민석, 김정의, 육현섭, 조란 )



## 개요

영어 원어민이 말하는 영상에서 원 발화자의 목소리 대신 사용자의 목소리를 입혀서 들려준다. 쉐도잉 기법(따라 말하기)은 원 발화자와 사용자의 성별과 목소리 톤이 다른 경우 똑같이 따라하는 것과 연습 후에 올바르게 따라했는지 비교하는 것에 어려움이 있다. 이에 따라 원 발화자의 말에 사용자의 목소리만 입혀서 원래의 발음과 속도, 억양은 그대로 들려주면서 사용자의 목소리를 통해서 정답 문장을 들려주기 때문에 말할 때도 더 쉽게 특징을 인식하고, 연습 후에도 정확한 비교도 할 수 있다.

또한 한 문장에서도 chunk 단위로 분리하여 길거나 어려운 문장인 경우 나눠서 학습할 수 있도록 한다.



## 제공 기능

1. 원어민의 말을 사용자의 목소리로 변환

2. 문장을 따라 말한 후 일치율 분석

3. 문장 혹은 청크 단위로 나누어 제공

   

## 음성 일치율 평가 기준

1. 발음 명확도(맞춘 단어)
2. 강세(음량)
3. 말하기 속도
4. 음 높낮이(인토네이션)



## 데이터 소개 및 전처리

1. Zerospeech 2019 데이터 사용
2. 음성데이터에서 mfcc, mel, tonnetz을 추출하여 xgboost모델 생성
3. mel을 이용해 VQ-VAE모델 훈련



		## 학습방법

1. xgboost : 추출된 특징들을 grid search(n_estimators, max_depth 조정)하여 모델 생성

2. VQ-VAE : mel값들을 모델에 넣은 후 reconstruct loss, vq- loss, perplexity 를 계산하여 훈련



	## 	Vq-vae 학습 방법

​	1. zerospeech 폴더로 경로 지정

​	2. 전처리: python preprocess.py in_dir=./ dataset=english

​	3. 학습: python train.py dataset=english	 



	## 	XGBoost 모델 생성 방법

​	1.전처리, 학습: evaluate.train('모델명')

​    2.분류 예측: user_in_speaker('사용자 아이디', '오디오파일')



## 	음성변환 Flow

 	1.102명의 원어민 목소리를 VQ-VAE모델에 학습

​	2. xgboost를 이용해 102명 중 사용자와 비슷한 목소리 선별

​	3. 비슷한 목소리 위치에 사용자 목소리를 더해서 VQ-VAE모델에 추가학습	

​	4. ted영상의 음성으로 음성 변환 실행



   ##      참조

* https://github.com/bshall/ZeroSpeech
* https://github.com/Raman-Raje/voiceClassifier