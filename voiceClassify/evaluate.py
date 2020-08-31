
from voiceClassify.utils import *
import numpy as np

from xgboost import XGBClassifier

import wave

from pickle import dump

from sklearn.preprocessing import normalize

from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")



def features():

    file_paths = []
    labels = []
    features = []
    rootdir = "./data"
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav'):
                fp = subdir+'/'+file
                with wave.open(fp,'r') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                if duration <=1:

                    pass
                else:
                    fp = subdir+'/'+file
                    features.append(extract_features(fp))
                    file_paths.append(fp)
                    labels.append(fp.split('/')[-2])
                    

    feat = np.array(features)

    return file_paths,labels,feat


def create_df():

    file_paths,labels,feat = features() # mfcc,mel,tonnetz 추출


    df = pd.DataFrame({"filepath":file_paths,"label":labels})
    df["mfccs"] = feat[:,0]

    df["mel"] = feat[:,1]

    df["tonnetz"] = feat[:,2]

    df["Final_label"] = feat[:,3]

    return df


def get_features(): # 추출한 특징들 묶기

    df = create_df()

    mfccs = np.vstack((df["mfccs"].values))

    mel = np.vstack((df["mel"].values))
    tonnetz = np.vstack((df["tonnetz"].values))


    labels = df["label"].values

    print("Features extracted successfully...")


    mfccs = normalize(mfccs,axis=0)

    mel = normalize(mel,axis=0)

    tonnetz = normalize(tonnetz,axis=0)
    df.to_csv('./preprocess_no_chroma_contrast.csv',index=False)
    return mfccs,mel,tonnetz,labels 


def train(modelname): # xgboost모델 만들기

    print("Making data set ready for classifier...")
    
    mfccs,mel,tonnetz,labels = get_features()

    xgb_x = np.hstack((mfccs,mel,tonnetz))
    xgb_y = labels

    print("*"*60)
    print(mfccs.shape)

    print(mel.shape)
    print(tonnetz.shape)

    print(xgb_x.shape)
    print(xgb_y.shape)



    X_train, X_test, y_train, y_test = train_test_split(xgb_x,xgb_y, test_size=0.25, random_state=42)

    kfold =StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
   
    param_grid = {'max_depth': [5,10,15], 'n_estimators': [50,100, 150]}
    grid = GridSearchCV(XGBClassifier(), param_grid, cv=kfold ,n_jobs=-1)
  
    grid.fit(X_train,y_train)
    with open("./model/clf-"+modelname+".pkl","wb") as f:
     
        dump(grid.best_estimator_,f)
        print("Model saved successfully...")   
 
    print("Accuracy for train ",accuracy_score(y_train,grid.predict(X_train)))
    print("Accuracy for test ",accuracy_score(y_test,grid.predict(X_test)))



if __name__ == "__main__":

    samples = eval(input("Please provide a list of voice samples\n"))
    assert type(samples) == list , "please provide list of paths"

    try:

        train(samples)
        
    except Exception as e:

        
        print(e)


         

