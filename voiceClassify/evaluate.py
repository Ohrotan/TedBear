import os
import shutil
from utils import *
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import wave
from ast import literal_eval
from pickle import load,dump
#from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,accuracy_score

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
#                    print(file+'passed')
                    pass
                else:
                    fp = subdir+'/'+file
                    features.append(extract_features(fp))
                    file_paths.append(fp)
                    labels.append(fp.split('/')[-2])
                    
#                    now = time.gmtime(time.time())
#                    print('hour:'+str(now.tm_hour-3),'min:'+str(now.tm_min), 'sec'+str(now.tm_sec))
#                    print(file+'feature extracted')
#                    print('*****')

    # convert feature into array
    feat = np.array(features)

    return file_paths,labels,feat


def create_df():

    file_paths,labels,feat = features()

    # creating the dataframe
    df = pd.DataFrame({"filepath":file_paths,"label":labels})
    df["mfccs"] = feat[:,0]
    #df["chroma"] = feat[:,1]
    #df["mel"] = feat[:,2]
    df["mel"] = feat[:,1]
    #df["contrast"] = feat[:,3]
    #df["tonnetz"] = feat[:,4]
    df["tonnetz"] = feat[:,2]
    #df["Final_label"] = feat[:,5]
    df["Final_label"] = feat[:,3]

    return df


def get_features():

    df = create_df()
    # stacking all samples to create a matrix
    mfccs = np.vstack((df["mfccs"].values))
    #chroma = np.vstack((df["chroma"].values))
    mel = np.vstack((df["mel"].values))
    tonnetz = np.vstack((df["tonnetz"].values))
    #contrast = np.vstack((df["contrast"].values))

    labels = df["label"].values

    print("Features extracted successfully...")


    # normalizing the features
    mfccs = normalize(mfccs,axis=0)
    #chroma = normalize(chroma,axis=0)
    mel = normalize(mel,axis=0)
    #contrast = normalize(contrast,axis=0)
    tonnetz = normalize(tonnetz,axis=0)
    df.to_csv('./preprocess_no_chroma_contrast.csv',index=False)
    return mfccs,mel,tonnetz,labels #,chroma,contrast


def train(modelname):

    print("Making data set ready for classifier...")
    
    #voiceSamples(samples)
    #data=pd.read_csv('preprocess_real.csv')
    #mfccs=np.vstack(pd.eval(df["mfccs"]))
    #chroma = np.vstack(pd.eval(df["chroma"]))
    #mel = np.vstack(pd.eval(df["mel"]))
    #tonnetz = np.vstack(pd.eval(df["tonnetz"]))
    #contrast = np.vstack(pd.eval(df["contrast"]))
    #mfccs=np.array(data["mfccs"].map(literal_eval).tolist()).reshape(3002,40)
    #contrast=np.array(data["contrast"].map(literal_eval).tolist()).reshape(3002, 7)
    #mel=np.array(data["mel"].map(literal_eval).tolist()).reshape(3002, 128)
    #chroma=np.array(data["chroma"].map(literal_eval).tolist()).reshape(3002,12)   
    #tonnetz=np.array(data["tonnetz"].map(literal_eval).tolist()).reshape(3002,6)
    #labels=np.array(data["label"])
    
    mfccs,mel,tonnetz,labels = get_features()
    #xgb_x =pd.concat([mfccs,contrast,mel,chroma,tonnetz], axis=1)
    xgb_x = np.hstack((mfccs,mel,tonnetz))
    xgb_y = labels

    # printing the necessary information
    print("*"*60)
    print(mfccs.shape)
    #print(chroma.shape)
    print(mel.shape)
    print(tonnetz.shape)
    #print(contrast.shape)
    print(xgb_x.shape)
    print(xgb_y.shape)


    ############## Training ###########
    X_train, X_test, y_train, y_test = train_test_split(xgb_x,xgb_y, test_size=0.25, random_state=42)
    '''
    for i in c:
        clf = svm.SVC(C = i,kernel='rbf', class_weight='balanced',probability=True)
        clf.fit(X_train,y_train)
        print("*"*60)
        print("For C = ",i)
        print("Accuracy for train ",accuracy_score(y_train,clf.predict(X_train)))
        print("Accuracy for test ",accuracy_score(y_test,clf.predict(X_test)))



    print(clf.classes_)
    c_ = eval(input("Please provide value of c for final..\n"))
    clf = svm.SVC(C = c_,kernel='rbf',class_weight="balanced",probability=True)
    '''
    kfold =StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    #param_grid = {'max_depth': [1, 5,10], 'n_estimators': [50, 100, 200,300] ,'learning_rate':[1, 0.1, 0.01],'min_child_weight':[1,3,5]}
    param_grid = {'max_depth': [5,10], 'n_estimators': [50,100]}
    grid = GridSearchCV(XGBClassifier(), param_grid, cv=kfold ,n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=100,oob_score=True,random_state=1534)
    #clf=XGBClassifier()
    #clf.fit(X_train,y_train)
    grid.fit(X_train,y_train)
    with open("./model/clf-"+modelname+".pkl","wb") as f:
        #dump(clf,f)
        dump(grid.best_estimator_,f)
        print("Model saved successfully...")    # saving the model
    #os.mkdir("model")ㄴ
    print("Accuracy for train ",accuracy_score(y_train,grid.predict(X_train)))
    print("Accuracy for test ",accuracy_score(y_test,grid.predict(X_test)))
    #print("Accuracy for train ",accuracy_score(y_train,clf.predict(X_train)))
    #print("Accuracy for test ",accuracy_score(y_test,clf.predict(X_test)))
    #print('final params',grid.best_params_)   # 최적의 파라미터 값 출력
    #print('best score', grid.best_score_)      # 최고의 점수
    #with open("./model/clf-"+modelname+".pkl","wb") as f:
    #    dump(clf,f)
    #    #dump(grid.best_estimator_,f)
    #    print("Model saved successfully...")
    
    #if os.path.exists("./data/"):
        #shutil.rmtree("./data/")    



if __name__ == "__main__":

    samples = eval(input("Please provide a list of voice samples\n"))
    assert type(samples) == list , "please provide list of paths"

    try:

        train(samples)
        
        #if os.path.exists("./data/"):
        #    shutil.rmtree("./data/")    
    except Exception as e:

        #if os.path.exists("./data/"):
        #    shutil.rmtree("./data/")    

        print(e)
        #print("Cleaning done...")

         

