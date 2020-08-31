import os
import librosa
import numpy as np



def extract_features(file_name,label = None): #mfcc, mel, tonnetz 추출
  
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    
    
    mfccs = np.expand_dims(mfccs,axis=0)

    mel = np.expand_dims(mel,axis=0)

    tonnetz = np.expand_dims(tonnetz,axis=0)

        
    if label:

        label = file_name.split('/')[-2]

    return mfccs,  mel, tonnetz, label 



def preprocess_voice(fp): #무음제거 등 특징추출할 파일들 고치기
    

    fname = fp.split("/")[-1].split(".")[0]
    
    if not os.path.exists("./data/{}".format(fname)):
        os.makedirs("./data/{}".format(fname))     
    
  
    os.system("ffmpeg -i {} -b:a 320000 ./data/{}/{}.wav".format(fp,fname,fname))
  
    os.system("sox ./data/{}/{}.wav ./data/{}/{}_sl.wav silence -l 1 0.1 1% -1 2.0 1%".format(fname,fname,fname,fname))
   
    os.system("ffmpeg -i ./data/{}/{}.wav -f segment -segment_time 5 -c copy ./data/{}/{}_%03d.wav".format(fname,fname,fname,fname))

    os.remove("./data/{}/{}.wav".format(fname,fname))
    os.remove("./data/{}/{}_sl.wav".format(fname,fname))


def voiceSamples(samples):
    

    for ep in samples:
        preprocess_voice(ep)




    
