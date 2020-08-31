import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize



def extract_features(file_name,label = None):
    
    """ 
    filename to extract the feature
    :type file_name:
    :param file_name: file path

    :type label:
    :param label:

    :raises:

    :rtype: returns the extrated features
    """
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    
    
    mfccs = np.expand_dims(mfccs,axis=0)
    #chroma = np.expand_dims(chroma,axis=0)
    mel = np.expand_dims(mel,axis=0)
    #contrast = np.expand_dims(contrast,axis=0)
    tonnetz = np.expand_dims(tonnetz,axis=0)

        
    if label:
        # We add also the classes of each file as a label at the end
        label = file_name.split('/')[-2]

    return mfccs,  mel, tonnetz, label #, chroma,contrast



def preprocess_voice(fp):
    
    """ 
    This function preprocess voice sample. The preprocessing is done as below.
    - Convert the audio file to k .wav format
    - Trim silence from the both side of audio file.
    - Split WAV file in 5 second segments

    :type fp: file_path (string)
    :param fp: file Path

    :raises: None

    :rtype: None
    """

    fname = fp.split("/")[-1].split(".")[0]
    
    if not os.path.exists("./data/{}".format(fname)):
        os.makedirs("./data/{}".format(fname))     
    
    # convert audio file to .wav format
    os.system("ffmpeg -i {} -b:a 320000 ./data/{}/{}.wav".format(fp,fname,fname))
    # trim the silence present in the file
    os.system("sox ./data/{}/{}.wav ./data/{}/{}_sl.wav silence -l 1 0.1 1% -1 2.0 1%".format(fname,fname,fname,fname))
    # break down  the audio file into 5 sec 
    os.system("ffmpeg -i ./data/{}/{}.wav -f segment -segment_time 5 -c copy ./data/{}/{}_%03d.wav".format(fname,fname,fname,fname))
    
    # remmooving the unnecessary files
    os.remove("./data/{}/{}.wav".format(fname,fname))
    os.remove("./data/{}/{}_sl.wav".format(fname,fname))


def voiceSamples(samples):
    
    """ 
    This function creates dataset for classifier.
    
    :type samplels: list or tuples
    :param samplels: give list of voice samples in a list or tuple format

    :raises:None

    :rtype:None
    """

    for ep in samples:
        preprocess_voice(ep)




    
