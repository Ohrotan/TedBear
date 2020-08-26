#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import IPython.display
from scipy.ndimage import gaussian_filter1d
import pysptk
from scipy.io import wavfile
import warnings
from pydub import AudioSegment
import ffmpeg
warnings.filterwarnings("ignore")
import speech_recognition as sr
from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))
import pysptk
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")
import copy


# In[40]:


import speech_recognition as sr9
r = sr9.Recognizer()
import nltk
#nltk.download('punkt')  # 처음한번 필요
from nltk.tokenize import word_tokenize

def get_word_error_rate(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return 100-result


# In[41]:


import sys
def editDistance(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def alignedPrint(list, r, h, result):
    print("Speecher words:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" "),
    print("\nShadowed words:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print("\ntypes of wrong:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
    print("\nCorrect Rate: " + result)

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    result=round(100-result,2)
    result_str = str("%.2f" % result) + "%"
    alignedPrint(list, r, h,result_str)
    print('D: 놓친단어, S: 잘못말한 단어, I: 원본에 없는 단어')
    return  result


# In[15]:


def preprocess(ted_audio_path,user_audio_path,png_save_path):
    ted,sr=librosa.load(ted_audio_path,sr=16000) # ted는 amplitude ,sr은 sample rate 16000고정

    abs_ted=abs(ted) # amplitude에 절대값취하기
    max_values_in_width=[]  # value = 한 구간(width) 에서의 최고값이 들어갈자리
    loc_of_max_values_in_width=[]   # loc 은 value의 index를 나타내는 값
    c=0
    i=0
    width_ted=int(len(ted)/16000)*900  # width_ted는 정보압축위해 단순화 할 구간 길이
    abs_ted=list(abs_ted ) 
    while i < len(range(len(ted))):
        t=abs_ted[i:i+width_ted]       #t = 0부터 y길이까지 구간으로 자른 것
        a=max(t)                 #  t 구간에서 가장큰값이 a에 저장됨
        b=t.index(a)             #   b= a값의 위치를 나타냄
        max_values_in_width.append(a)            # value리스트에 값저장 
        loc_of_max_values_in_width.append(i+b)            # loc에도 위치저장
        if i+width_ted > len(abs_ted):   # 구간이 y길이보다 넘어가면 취소
            break
        else:
            c=c+1                  # y길이 안넘어가면 반복
            i=width_ted*c
    empty_array_ted=np.empty(len(abs_ted))      # empty_array_ted는 빈 array 이제 아까 추가했던 값들을 바탕으로  정보압축한 그래프 생성
    empty_array_ted.fill(np.nan)
    for i in range(len(loc_of_max_values_in_width)):       
        location=loc_of_max_values_in_width[i]
        empty_array_ted[location]=max_values_in_width[i]     # 아까 최대값 이었던 부분 대입 
    first_local_maximum_ted=np.where(empty_array_ted>0.15)[0][0]      #뾰족뾰족 솟은 부분에서 0.15 보다 처음으로 높았던 부분 을 뽑음 (뒤에서필요) 
    df=pd.DataFrame(empty_array_ted)                               
    df.interpolate(method='polynomial',order=2,inplace=True) # 이거는 직선 매끄럽게 깍아주려고 2차함수꼴로 바꿔줌
    df_list=list(df[0])        
    after_interpolate=list(map(lambda x: 0 if x<0 else x,df_list))  # 요거는 2차함수꼴로 깍아주다보면 -값이 발생해서 그걸 0으로바꿔주는과정
    df[0]=after_interpolate
    df.fillna(0,inplace=True)

# ###############위에과정반복###########
    you,sr1=librosa.load(user_audio_path,sr=16000)
    abs_you=abs(you)
    width_you=int(len(you)/16000)*900
    max_values_in_width_1=[]
    loc_of_max_values_in_width_1=[]
    c1=0
    i=0
    abs_you=list(abs_you)
    while i < len(range(len(you))):
        t1=abs_you[i:i+width_you]
        a1=max(t1)
        b1=t1.index(a1)
        max_values_in_width_1.append(a1)
        loc_of_max_values_in_width_1.append(i+b1)
        if i+width_you > len(abs_you):
            break
        else:
            c1=c1+1
            i=width_you*c1
    normalize=   (max(abs_ted)/max(abs_you))*0.95            ## 두 음성데이터의 크기 맞춰주기위해 최대값기준으로 맞춰줌
    after_normalize=list(map(lambda x: normalize*x, max_values_in_width_1))         
    empty_array_you=np.empty(len(abs_you))
    empty_array_you.fill(np.nan)       
    for i in range(len(loc_of_max_values_in_width_1)):
        location1=loc_of_max_values_in_width_1[i]
        empty_array_you[location1]=after_normalize[i]
    first_local_maximum_you=np.where(empty_array_you>0.15)[0][0]
    df1=pd.DataFrame(empty_array_you)        
    df1.interpolate(method='polynomial',order=2,inplace=True)
    df_list_you=list(df1[0])
    after_interpolate1=list(map(lambda x: 0 if x<0 else x,df_list_you))
    df1[0]=after_interpolate1
    df1.fillna(0,inplace=True)
    return first_local_maximum_ted, first_local_maximum_you,df,df1


# In[22]:


def eval_speed(first_local_maximum_ted, first_local_maximum_you,df,df1):
    if first_local_maximum_you>first_local_maximum_ted:                     ## A음성(first1)이 B음성(first)보다 음성이 늦게 시작한다면  
        diff= first_local_maximum_you-first_local_maximum_ted              # 두 음성 시작점 차이 구해가지고
        blue_ted=list(df[0])  # blue가 테드
        orange_you=list(df1[0])# orange가 나
        add=[0 for z in range(diff)]
        blue_ted_added=add+blue_ted                 #  더해줘서 blue1생김
        df_ted_added=pd.DataFrame(blue_ted_added)
    # 점수 짧은걸써야함 first1부터 df1[0]       ############# 이거는 점수구하는 과정###########
        if len(df_ted_added[0])>len(df1[0]):
            index_diff=((len(df_ted_added[0])-(len(df1[0]))))
            time_diff=index_diff/16000
            time_diff_per=index_diff/(len(df_ted_added[0])-first_local_maximum_you)
            time_result=round((1-time_diff_per)*100,1)
        else:
            index_diff=((len(df1[0]))-(len(df_ted_added[0])))
            time_diff=index_diff/16000
            time_diff_per=index_diff/(len(df_ted_added[0])-first_local_maximum_you)
            time_result=round((1+time_diff_per)*100,3)   
    #테듣가 더길떄               ######## 방금 위에한거의 반대과정 A보다 B가 늦게시작할경우######
    elif first_local_maximum_you<=first_local_maximum_ted:
        diff=first_local_maximum_ted-first_local_maximum_you
        blue_ted=list(df[0])
        orange_you=list(df1[0])
        add=[0 for z in range(diff)]
        orange_you_added=add+orange_you # 내가짧은이까 더해줘서 orange1이생김
        df_you_added=pd.DataFrame(orange_you_added)
        # 테드가 첨부분은 ㄴ더 나중에나옴 
        if len(df_you_added[0])>len(df[0]):
            index_diff=((len(df_you_added[0])-(len(df[0]))))
            time_diff=index_diff/16000
            time_diff_per=index_diff/(len(df[0])-first_local_maximum_ted)
            time_result=round((1+time_diff_per)*100,1)
        else:
            index_diff=((len(df[0]))-(len(df_you_added[0])))
            time_diff=index_diff/16000
            time_diff_per=index_diff/(len(df[0])-first_local_maximum_ted)
            time_result=round((1-time_diff_per)*100,1)
            print(time_diff,time_result)
    return time_diff, time_result


# In[34]:


def eval_strength(first_local_maximum_ted, first_local_maximum_you,df,df1):
    ## 이제 다 좌표찍는걸 그리는과정###                                    # 두음성 시작점 맞춰주는 과정
    if first_local_maximum_you>first_local_maximum_ted:                     ## A음성(first1)이 B음성(first)보다 음성이 늦게 시작한다면  
        diff= first_local_maximum_you-first_local_maximum_ted              # 두 음성 시작점 차이 구해가지고
        blue_ted=list(df[0])  # blue가 테드
        orange_you=list(df1[0])# orange가 나
        add=[0 for z in range(diff)]
        blue_ted_added=add+blue_ted                 #  더해줘서 blue1생김
        df_ted_added=pd.DataFrame(blue_ted_added)
        plt.figure(figsize=(20,8))
        # 점수 짧은걸써야함 first1부터 df1[0]       ############# 이거는 점수구하는 과정###########
        if len(df_ted_added[0])>len(df1[0]):
            area =[]
            for z in range(first_local_maximum_you,len(df1[0]),1):
                area.append(df_ted_added[0][z])        
            points=1-(sum(abs(df1[0][first_local_maximum_you:len(df1[0])]-df_ted_added[0][first_local_maximum_you:len(df1[0])]))/sum(area))
            #points1=np.correlate(  df_ted_added[0] [first_local_maximum_you:len(df1[0])],df1[0][first_local_maximum_you:len(df1[0])])
            points2=np.mean(np.correlate(  df_ted_added[0] [first_local_maximum_you:len(df1[0])],df1[0][first_local_maximum_you:len(df1[0])],'full'))
            points3=cos_sim(df_ted_added[0] [first_local_maximum_you:len(df1[0])],df1[0][first_local_maximum_you:len(df1[0])])
            
        else:
            area =[]
            for z in range(first_local_maximum_you,len(df_ted_added[0]),1):
                area.append(df_ted_added[0][z])        
            points=1-(sum(abs(df1[0][first_local_maximum_you:len(df_ted_added[0])]-df_ted_added[0][first_local_maximum_you:len(df_ted_added[0])]))/sum(area))
            #points1=np.correlate(  df_ted_added[0] [first_local_maximum_you:len(df1[0])],df1[0][first_local_maximum_you:len(df1[0])])
            points2=np.mean(np.correlate(  df_ted_added[0] [first_local_maximum_you:len(df_ted_added[0])],df1[0][first_local_maximum_you:len(df_ted_added[0])],'full'))
            points3=cos_sim(df_ted_added[0] [first_local_maximum_you:len(df_ted_added[0])],df1[0][first_local_maximum_you:len(df_ted_added[0])]  )
            ##############################
        your_voice = gaussian_filter1d(df1[0], sigma=2)      # 이거도 뾰족부분 깍는과정인데 왜 두번들어가더라#
        line1,=plt.plot(your_voice,color='orange',linewidth=5)
        line2,=plt.plot(df_ted_added[0],color='blue',linewidth=5)
        plt.title('Strength Result',fontsize=50)
        plt.legend(handles=(line1,line2),labels=('Ted','You'),fontsize=20)
        plt.ylabel('Strength',fontsize=20)
        plt.tick_params(axis='x', which='both',bottom=False,top=False,labelbottom=False)
        plt.show()
        plt.savefig(png_save_path+'strength_result.png')
    #테듣가 더길떄               ######## 방금 위에한거의 반대과정 A보다 B가 늦게시작할경우######
    elif first_local_maximum_you<=first_local_maximum_ted:
        diff=first_local_maximum_ted-first_local_maximum_you
        blue_ted=list(df[0])
        orange_you=list(df1[0])
        add=[0 for z in range(diff)]
        orange_you_added=add+orange_you # 내가짧은이까 더해줘서 orange1이생김
        df_you_added=pd.DataFrame(orange_you_added)
        plt.figure(figsize=(20,8));
        # 테드가 첨부분은 ㄴ더 나중에나옴 
        if len(df_you_added[0])>len(df[0]):
            area =[]
            for z in range(first_local_maximum_ted,len(df[0]),1):
                area.append(df[0][z])
            points=1-(sum(abs(df[0][first_local_maximum_ted:len(df[0])]-df_you_added[0][first_local_maximum_ted:len(df[0])]))/sum(area))
            #points1=np.correlate(df_you_added[0][first_local_maximum_ted:len(df[0])],df[0][first_local_maximum_ted:len(df[0])])
            points2=np.mean(np.correlate(df_you_added[0][first_local_maximum_ted:len(df[0])],df[0][first_local_maximum_ted:len(df[0])],'full'))
            points3=cos_sim( df_you_added[0][first_local_maximum_ted:len(df[0])],df[0][first_local_maximum_ted:len(df[0])] )
        else:
            area =[]
            for z in range(first_local_maximum_ted,len(df_you_added[0]),1):
                    area.append(df[0][z])
            points=1-(sum(abs(df1[0][first_local_maximum_ted:len(df_you_added[0])]-df_you_added[0][first_local_maximum_ted:len(df_you_added[0])]))/sum(area))
            #points1=np.correlate(df_you_added[0][first_local_maximum_ted:len(df_you_added[0])],df[0][first_local_maximum_ted:len(df_you_added[0])])
            points2=np.mean(np.correlate(df_you_added[0][first_local_maximum_ted:len(df_you_added[0])],df[0][first_local_maximum_ted:len(df_you_added[0])],'full'))
            points3=cos_sim( df_you_added[0][first_local_maximum_ted:len(df_you_added[0])],df[0][first_local_maximum_ted:len(df_you_added[0])] )
        
        line1,=plt.plot(df[0],color='blue',linewidth=5)  
        line2,=plt.plot(df_you_added[0],color='orange',linewidth=5)
        plt.title('Strength Result',fontsize=50)
        plt.legend(handles=(line1,line2),labels=('You','Ted'),fontsize=20)
        plt.ylabel('Strength',fontsize=20)
        plt.tick_params(axis='x', which='both',bottom=False,top=False,labelbottom=False)
        plt.show()
        plt.savefig(png_save_path+'strength_result.png')
    result1=int(points*100)
    result2=points2
    result3=points3
    return result1, result2, result3


# In[35]:


def eval_pitch(ted_audio_path,user_audio_path,png_save_path):
    sr, x = wavfile.read(ted_audio_path)
    assert sr == 16000
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 80
    f_you = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")

    sr1, x1 = wavfile.read(user_audio_path)
    assert sr1 == 16000
    x1 = x1.astype(np.float64)
    frame_length = 1024
    hop_length = 80

    # F0 estimation
    f_ted= pysptk.swipe(x1.astype(np.float64), fs=sr1, hopsize=hop_length, min=60, max=240, otype="f0")
    plt.figure(figsize=(20,5))
    ##############
    width=int(len(f_ted)/22)
    width1=int(len(f_you)/22)
    if np.where(f_you>=60)[0][0] > np.where(f_ted>=60)[0][0]: #테드가 왼쪽에서 더 빠르게 시작하면?
        diff=np.where(f_you>=60)[0][0]-np.where(f_ted>=60)[0][0]
        zero_=np.zeros(diff)
        new_f0=np.r_[zero_,f_ted]
        cal0=copy.copy(new_f0)
        cal0[np.where(cal0<75)]=0
        new_f0[np.where(new_f0<75)]=0
        value=[]
        loc=[]
        c3=0
        i=0
        new_f0=list(new_f0)
        while i < len(range(len(new_f0))):
            ten=new_f0[i:i+width]
            a=max(ten)
            b=ten.index(a)
            value.append(a)
            loc.append(i+b)
            if i+width > len(new_f0):
                break
            else:
                c3=c3+1
                i=width*c3
        base=np.empty(len(new_f0))
        base.fill(np.nan)       
        for i in range(len(loc)):
            location=loc[i]
            base[location]=value[i]
        df_green=pd.DataFrame(base)
        plt.figure(figsize=(20,8))
        df_green.interpolate(method='polynomial',order=2,linewidth=2,inplace=True)

        bbb=list(df_green[0])
        ccc=list(map(lambda x: 0 if x<0 else x,bbb))
        df_green[0]=ccc
        df_green.fillna(0,inplace=True)
        for g in range(len(df_green)):
            if df_green[0][g]>max(f_ted)*1.2:
                df_green[0][g]=max(f_ted)*1.2
        #합친게 you0
        you0=f_you
        #you0=np.r_[f_you,np.zeros(len(new_f0)-len(f_you))]
        cal1=copy.copy(you0)
        cal1[np.where(you0<75)]=0
        you0[np.where(you0<75)]=0
        value1=[]
        loc1=[]
        c1=0
        i1=0
        you0=list(you0)
        while i < len(range(len(you0))):
            ten1=you0[i1:i1+width1]
            a1=max(ten1)
            b1=ten1.index(a1)
            value1.append(a1)
            loc1.append(i1+b1)
            if i1+width1 > len(you0):
                break
            else:
                c1=c1+1
                i1=width1*c1
        base1=np.empty(len(you0))
        base1.fill(np.nan)       
        for i in range(len(loc1)):
            location1=loc1[i]
            base1[location1]=value1[i]
        df_coral=pd.DataFrame(base1)
        df_coral.interpolate(method='polynomial',order=2,linewidth=2,inplace=True)
        bbb1=list(df_coral[0])
        ccc1=list(map(lambda x: 0 if x<0 else x,bbb1))
        df_coral[0]=ccc1
        df_coral.fillna(0,inplace=True)
        for h in range(len(df_coral)):
            if df_coral[0][h]>max(f_you)*1.2:
                df_coral[0][h]=max(f_you)*1.2
        df_coral[0]=df_coral[0]*max(f_ted)/max(f_you)

    else:   # 테드가 더늦겟시작
        diff=np.where(f_ted>=60)[0][0]-np.where(f_you>=60)[0][0]
       
        zero_ted=np.zeros(diff)
        
        new_ted=np.r_[zero_ted,f_you]
        cal_ted=copy.copy(new_ted)
        cal_ted[np.where(cal_ted<75)]=0
        new_ted[np.where(new_ted<75)]=0
        value1=[]
        loc1=[]
        c1=0
        i1=0
        new_ted=list(new_ted)
        while i1 < len(range(len(new_ted))):
            ten1=new_ted[i1:i1+width1]
            a1=max(ten1)
            b1=ten1.index(a1)
            value1.append(a1)
            loc1.append(i1+b1)
            if i1+ width1  > len(new_ted):
                break
            else:
                c1=c1+1
                i1=width1*c1
        base=np.empty(len(new_ted))
        base.fill(np.nan)
        for h in range(len(loc1)):
            location1=loc1[h]
            base[location1]=value1[h]
        df_green=pd.DataFrame(base)
        df_green.interpolate(method='polynomial',order=2,linewidth=2,inplace=True)
        bbb=list(df_green[0])
        #print(bbb)
        ccc=list(map(lambda x: 0 if x<0 else x ,bbb))
        df_green[0]=ccc
        df_green.fillna(0,inplace=True)
        for g in range(len(df_green)):
            if df_green[0][g]>max(f_you)*1.2:
                df_green[0][g]=max(f_you)*1.2  
        f_ted1=f_ted
        #f_ted1=np.r_[f_ted,np.zeros(abs(len(new_ted)-len(f_ted)))]
        cal2=copy.copy(f_ted1)
        cal2[np.where(f_ted1<75)]=0
        f_ted1[np.where(f_ted1<75)]=0
        value2=[]
        loc2=[]
        c2=0
        i2=0
        f_ted1=list(f_ted1)
        while i2 <len(range(len(f_ted1))):
            ten2=f_ted1[i2:i2+width]
            a2=max(ten2)
            b2=ten2.index(a2)
            value2.append(a2)
            loc2.append(i2+b2)
            if i2+width >len(f_ted1):
                break
            else:
                c2=c2+1
                i2=width*c2
        base2=np.empty(len(f_ted1))
        base2.fill(np.nan)
        for i in range(len(loc2)):
            location2= loc2[i]
            base2[location2]=value2[i]
        df_coral=pd.DataFrame(base2)
        df_coral.interpolate(method='polynomial',order=2,linewidth=2,inplace=True)
        bbb2=list(df_coral[0])
        ccc2=list(map(lambda x:0 if x <0 else x , bbb2))
        df_coral[0]=ccc2
        df_coral.fillna(0,inplace=True)
        for g in range(len(df_coral)):
            if df_coral[0][g]>max(f_ted)*1.2:
                df_coral[0][g]=max(f_ted)*1.2

    plt.plot(df_green,color='mediumseagreen',linewidth=5)
    plt.plot(df_coral,color='coral',linewidth=5) 
    plt.title('Pitch Result',fontsize=50)
    plt.legend(['Ted','You'],fontsize=20)
    plt.ylabel('Pitch',fontsize=20)
    plt.tick_params(axis='x', which='both',bottom=False,top=False,labelbottom=False)
    plt.savefig(png_save_path +'pitch_result.png')


# In[48]:


def eval_pronounciation(ted_audio_path,user_audio_path):
    r = sr9.Recognizer()
    with sr9.AudioFile(ted_audio_path) as source:
        audio = r.record(source)
        ted_answer=r.recognize_google(audio)
    answer=r.recognize_google(audio)
    answer=answer.lower()
    with sr9.AudioFile(user_audio_path) as source:
        audio = r.record(source)
        your_answer=r.recognize_google(audio)
    mine=r.recognize_google(audio)
    mine=mine.lower()
    answer_token=word_tokenize(answer)
    mine_token=word_tokenize(mine)
    get_word_error_rate(mine_token,answer_token)
    result=wer(mine_token,answer_token)
    return ted_answer, your_answer,result


# In[50]:


def eval(ted_audio_path,user_audio_path,png_save_path):
    first_local_maximum_you,first_local_maximum_ted,df,df1 = preprocess(ted_audio_path,user_audio_path,png_save_path)
    return  eval_speed(first_local_maximum_ted, first_local_maximum_you,df,df1),eval_strength(first_local_maximum_ted, first_local_maximum_you,df,df1),eval_pitch(ted_audio_path,user_audio_path,png_save_path),eval_pronounciation(ted_audio_path,user_audio_path)


# In[ ]:





# In[ ]:




