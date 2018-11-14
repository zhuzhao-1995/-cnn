# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:17:41 2018

@author: Administrator
"""

from __future__ import print_function
#from numpy.random import seed
#seed(3)  
#from tensorflow import set_random_seed
#set_random_seed(4)  
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input,Bidirectional
from keras.layers import Conv2D, AveragePooling2D,LSTM,Reshape,Flatten,Permute,AveragePooling3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from sklearn import preprocessing  
import pandas as pd
import numpy as np
import time
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

#导入各种程序包库
#from __future__ import print_function
import keras
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import utils as np_utils
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Convolution2D, MaxPooling2D
import csv
import numpy as np
import pandas as pd
import os
import sys


######## 设置参数:
batch_size = 10
epochs = 10
rate=0.2
e=1
########下载整理数据#############
print('Loading data...')
#OSError: Initializing from file failed  因为有中文名，应以下面例子改
#f = open('我的文件.csv')
#res = pd.read_csv(f)
f1_data= open('863-齿轮断齿-样本.csv')
f21_data= open('863-齿轮裂纹-样本.csv')
f31_data= open('863-大齿轮裂纹+小齿轮断齿（断一半）-样本.csv')
f41_data= open('863-轴承内圈故障-样本.csv')


c1_data=pd.read_csv(f1_data,header=None)
c21_data=pd.read_csv(f21_data,header=None)
c31_data=pd.read_csv(f31_data,header=None)
c41_data=pd.read_csv(f41_data,header=None)


c1_data=np.array(c1_data)
c21_data=np.array(c21_data)
c31_data=np.array(c31_data)
c41_data=np.array(c41_data)
#c1_y=np.array(c1_y)
#c4_y=np.array(c4_y)
#c6_y=np.array(c6_y)

#拼接各数组


x_data=np.vstack((c1_data,c21_data,c31_data,c41_data))

#给数据加上表签，0 =断齿 ；1 = 裂纹 ; 2 = 断齿裂纹  ； 3 = 内圈 ；
y1=np.zeros((450,1), dtype = np.int)
y2=np.ones((450,1), dtype = np.int)
y3=np.ones((450,1), dtype = np.int)
y3=y3*2
y4=np.ones((450,1), dtype = np.int)
y4=y4*3

y_label=np.vstack((y1,y2,y3,y4))

#划分数据集测试集
x = x_data
y = y_label
# print(X.shape,y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)
# print('X_train的数据形式',X_train.shape)
# print('X_test的数据形式',X_test.shape)
# print('y_train的数据形式',y_train.shape)
# print('y_test的数据形式',y_test.shape)
#把（900，52100）变成（900，10，5210）
x_train=x_train.reshape(1350,10,512)
x_test=x_test.reshape(450,10,512)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
##############################################改(0 0 0 1)(0 0 1 0)
y_train=np_utils.to_categorical(y_train,num_classes=4) 
y_test=np_utils.to_categorical(y_test,num_classes=4)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape) 

 

#########建立模型main_input（model）####
main_input = Input(shape= (10, 512), name='main_input')
print('Build model...')


x=LSTM(100,activation='tanh',recurrent_activation='sigmoid',return_sequences=True)(main_input)
x=BatchNormalization(axis=-1)(x)
x=LSTM(50,activation='tanh',recurrent_activation='sigmoid',return_sequences=True)(x)
x=BatchNormalization(axis=-1)(x)
x=LSTM(10,activation='tanh',recurrent_activation='sigmoid',return_sequences=True)(x)
x=BatchNormalization(axis=-1)(x)

x=Flatten()(x)
x=Dropout(rate, noise_shape=None, seed=None)(x)
x=Dense(10)(x)
x=BatchNormalization(axis=-1)(x)
x=Activation('sigmoid')(x)
#main_out=Dense(2,activation='softmax')(x)
main_out=Dense(4,activation='softmax')(x)
model = Model(main_input,main_out)

#model.summary()：打印出模型概况，它实际调用的是keras.utils.print_summary
#model.summary()
######编译训练##########

       
model.compile(loss='mse',optimizer='Adadelta',
              metrics=['accuracy'])

#t0 = time.time()
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
############计算训练时间#############
#fit_time = time.time() - t0
#print('fit_time:',fit_time)
#history_losses=history.history['loss']
#history_val_losses=history.history['val_loss']

score = model.evaluate(x_test, y_test, verbose=0)
print('score loss:', score[0])
print('score accuracy:', score[1])

