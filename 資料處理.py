#%%
# import pandas as pd
import numpy as np
import tensorflow as tf
import random
from pandas import DataFrame
import matplotlib.pyplot as plt
import keras
import re
# -*- coding: utf-8 -*-
from opencc import OpenCC
from sklearn.utils import shuffle
#%%
df = pd.read_excel('data0.xlsx')
df1 = pd.read_csv('news1.csv')

#%%
b=0
for i in range(30000,45000):
    a=len(str(df1.iloc[i,0]))
    if a>10 and a<50:
        b+=1
        new=pd.DataFrame({'txt':df1.iloc[i,0]},index=[1])
        df=df.append(new,ignore_index=True)
    if b >=1850:
        break
#%%
dfa = pd.read_excel('data0.xlsx')
dfb = pd.read_excel('data1.xlsx')
#%%
#去除非中文字串
def find_ch_en_num(file):
    pattern = re.compile("[^\u4e00-\u9fa5^]")
    chinese = re.sub(pattern,'', file)
    return chinese
def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern,'', file)
    return chinese
#%%
#刪除字數過低or高
def del_HoL_seq(df):
    low=[]
    low_n=[]
    for i in range(0,len(df)):
        if 7>len(str(df.iloc[i,0])):
            low.append(i)
            #print(df.iloc[i-1,0])       
    if low != []:
        low_n=list(low)
        low_n.sort(reverse=True)
        a=0
        for i in range(0,len(low_n)):
            a+=1
            df=df.drop([low_n[i]],axis = 0,inplace = False)
    return df
#刪去重複項
def del_same(df):
    num=len(df)
    repeat_n=[]
    for o in range(0,num-1):
        only=find_chinese(df.iloc[o,0])
        for r in range(o+1,num):
            repeat=find_chinese(df.iloc[r,0])
            #print(o,only,'\n',r,repeat,'\n')
            if only == repeat :
                #print(o,only,'\n',r,repeat,'\n')
                repeat_n.append(r)
    if repeat_n != []:
        repeat_n2=list(set(repeat_n))
        repeat_n2.sort(reverse=True)
        repeat_n2
        a=0
        for i in range(0,len(repeat_n2)):
            a+=1
            df=df.drop([repeat_n2[i]],axis = 0,inplace = False)
    return df
#%%繁簡轉換
#hk2s: 繁體中文 (香港) -> 簡體中文 #s2hk: 簡體中文 -> 繁體中文 (香港) #s2t: 簡體中文 -> 繁體中文 #s2tw: 簡體中文 -> 繁體中文 (台灣) #s2twp: 簡體中文 -> 繁體中文 (台灣, 包含慣用詞轉換) #t2hk: 繁體中文 -> 繁體中文 (香港) #t2s: 繁體中文 -> 簡體中文 #t2tw: 繁體中文 -> 繁體中文 (台灣) #tw2s: 繁體中文 (台灣) -> 簡體中文 #tw2sp: 繁體中文 (台灣) -> 簡體中文 (包含慣用詞轉換 )

cc = OpenCC('s2twp')
def s2tw(df):
    for i in range(0,len(df)):
        print(i)
        df.iloc[i,0]=cc.convert(df.iloc[i,0])
    return df
#%%
#%%
for i in range(0,len(df)):
    try:
        df.iloc[i,0]=find_chinese(df.iloc[i,0])
    except:
        df.iloc[i,0]=None
df=df.dropna(axis=0, how='any')



#%%
#刪除指定cell
#df=df.drop([150],axis = 0,inplace = False)
#%%
df = del_HoL_seq(df)
df = del_same(df)
df = s2tw(df)
df = shuffle(df)
df1 = del_HoL_seq(df1)
df1 = del_same(df1)
df1 = s2tw(df1)
df1 = shuffle(df1)
for i in range(11300,len(df)):
    df.iloc[i,1]=str(1)
for i in range(11300,len(df1)):
    df1.iloc[i,1]=str(0)


df.to_excel('data_0-22tw1_0105.xlsx',index=False,header=False)
df1.to_excel('news_0-22tw1_0105.xlsx',index=False,header=False)