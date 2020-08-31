import os
from os import walk
import shutil
import numpy as np
from utils import *
from pickle import load,dump
from sklearn.preprocessing import normalize


class VoiceClassify():

    """ The class for classifying voice samples. """

    def __init__(self,modelname,user_audio_file):

        # loading the model
        with open("./model/clf-"+modelname+".pkl","rb") as f:
            self.clf = load(f)
        self.user_audio_file=user_audio_file
        
    @staticmethod
    def _process(self):

        """ 
        This method is called internally to process the auido file.
        :type fp: file_path of sample file
        :param fp: file_path

        :raises: Raises an excetion if something went wrong.

        :rtype: Returns the numpy array fo features extracted from voice.
        """
        #fname = fp.split("/")[-1].split(".")[0]

        try:

            #if not os.path.exists("./temp"):
            #    os.mkdir("temp") 

            # convert audio file to .wav format
            #os.system("ffmpeg -i {} -b:a 320000 ./temp/{}.wav".format(fp,fname))
            # trim the silence present in the file
            #os.system("sox ./temp/{}.wav ./temp/{}.wav silence -l 1 0.1 1% -1 2.0 1%".format(fname,fname))

            #print("Audio preprocessed..")
#             flist = []
#             for (dirpath, dirnames, filenames) in walk("./test"):
#                 flist.extend(filenames)
#                 break
#             paths = [os.path.realpath("./test/{}".format(name)) for name in flist ]
            
            result =[]
            mfcc,  mel,  tonnetz,_ = extract_features(self.user_audio_file)
            re = np.hstack((normalize(mfccs),normalize(mel),normalize(tonnetz)))
            result.append(re)
             
#             for path in paths:
#                 mfccs,  mel,  tonnetz,_ = extract_features(path)
#                 re = np.hstack((normalize(mfccs),normalize(mel),normalize(tonnetz)))
#                 result.append(re)
            
            # remmooving the unnecessary files
            #shutil.rmtree("./temp/")
            return result

        except Exception as e:

            # remmooving the unnecessary files
            #shutil.rmtree("./temp/")
            #print("Cleaning done...")
            print(e)
            
    def classes_(self):

        """ 
        Returns the classes of the model
        :type self:
        :param self:
    
        :raises: None
    
        :rtype: classes_ object
        """    
        return self.clf.classes_

    def predict(self):

        """ 
        Predicts the class label of the file.
        :type self: 
        :param self:

        :type fp: absolute path of sample audio file
        :param fp: file path to be classified

        :raises: NA

        :rtype: predicted class and their probabilities and classes
        """
        dps = self._process(self)
        
        result = []
        for dp in dps:
            prediction,prob,classes = self.clf.predict(dp),self.clf.predict_proba(dp),self.clf.classes_
            result.append((prediction,prob,classes))
        
        return result



    
