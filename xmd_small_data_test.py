# Author: xumingda
#!/usr/bin/env python
# -*-  coding:utf-8 -*-
#正常的程序
#导入各种程序包库

###需要改的地方！！！！！： 文件目录地址，数据输入的样本大小，训练的批次轮次
#导入各种程序包库
from __future__ import print_function
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

#输入文件的地址，和每个数据样本包含的数据点个数num=1024=32x32,后面的四通道也是32x32
#  num ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
#输入文件的地址，和每个数据样本包含的数据点个数num
def init_data(csv_files,num):
    # df = pd.read_csv(csv_files,engine='python')
    data =[]
    with open(csv_files,encoding='utf-8') as csvfile:
        read_CSV = read_CSV = csv.reader(csvfile)
        for row in read_CSV:
            data.append(row)
        #print(row,i,data)
        data = np.array(data) #将列表转变成数组
        # print(data.shape)
        # print(type(data))
        #判断是否可以被num（样本长度）整除
        if data.shape[1]%num == 0:
            data=data.reshape(-1,num)
        else:
            #除法向下取整，判断一个csv文件之中能取多少个样本
            a=data.shape[1]//num
            # print(a)
            #对数据进行切片处理，取样本数的整数倍数据，后面的不要了
            data=data[:,:a*num]
            #对数据进行分行，形成数组，一行是一个样本
            if data.shape[1]%num == 0:
                data=data.reshape(-1,num)
    return data

### 读取文件夹下的所有csv文件
######os.listdir(r'D:/TJU/学习/GITHUB/18#右旋-151016-100908.csv')
#  父目录 可以更改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
parent_dir=r'D:\TJU\学习\GITHUB\test'
#  子目录  可以更改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
sub_dirs=r'good_csv',r'error_csv',r'temp'

#得到 n 个good.csv
def get_goodcsv_files(parent_dir,sub_dirs):
    csv_files = []
    for l, sub_dir in enumerate(sub_dirs):
        csv_path = os.path.join(parent_dir, sub_dir)
        for (dirpath, dirnames, filenames) in os.walk(csv_path):
            for filename in filenames:
                if filename.endswith('good.csv') :
                    filename_path = os.sep.join([dirpath, filename])
                    csv_files.append(filename_path)
    return csv_files

#得到 n 个wrong.csv
def get_wrongcsv_files(parent_dir,sub_dirs):
    csv_files = []
    for l, sub_dir in enumerate(sub_dirs):
        csv_path = os.path.join(parent_dir, sub_dir)
        for (dirpath, dirnames, filenames) in os.walk(csv_path):
            for filename in filenames:
                if filename.endswith('wrong.csv') :
                    filename_path = os.sep.join([dirpath, filename])
                    csv_files.append(filename_path)
    return csv_files

csv_goodfiles = get_goodcsv_files(parent_dir,sub_dirs)
csv_goodfiles =  np.array(csv_goodfiles)
# print(np.array(csv_goodfiles).shape)
# print(csv_goodfiles)

csv_wrongfiles = get_wrongcsv_files(parent_dir,sub_dirs)
csv_wrongfiles =np.array(csv_wrongfiles)
# print(csv_wrongfiles.shape)
# print(csv_wrongfiles)


#建立好的数据集正常的数据的数组a  #得到正常信号的数据集
# i=0
a=[]
#可以更改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
a=np.array(a).reshape(-1,1024)
for file in csv_goodfiles:
    b=init_data(file,1024)
    a=np.vstack((a,b))
    # i=i+1
    # print(a.shape,i)
#得到wrong的数据集，c故障数据的数组c
# i=0
c=[]
c=np.array(c).reshape(-1,1024)
for file in csv_wrongfiles:
    d=init_data(file,1024)
    c=np.vstack((c,d))
    # i=i+1
    # print(c.shape)

#将数据整合到一起
goodandwrong_data=np.vstack((a,c))
# print(goodandwrong_data.shape)
# print(goodandwrong_data)


#正常数据需要几行
a_lines = a.shape[0]
#故障数据需要几行
c_lines = c.shape[0]

#给数据加上表签，1 =正常 ；0 = 故障
y1=np.ones((a_lines,1), dtype = np.int)
y2=np.zeros((c_lines,1), dtype = np.int)
y_label=np.vstack((y1,y2))
# print(y_label.shape)

#划分数据集测试集
X = goodandwrong_data
y = y_label
# print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
# print('X_train的数据形式',X_train.shape)
# print('X_test的数据形式',X_test.shape)
# print('y_train的数据形式',y_train.shape)
# print('y_test的数据形式',y_test.shape)
# print(type(X_train))

##########32,32也许需要改的
X_train = X_train.reshape(-1,32,32,1)
X_test = X_test.reshape(-1,32,32,1)
#####标签转化成one_hot的形式
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test  = np_utils.to_categorical(y_test,  num_classes=2)
# print(y_train.shape)
# print(y_test.shape)
#设置随机种子
np.random.seed(1337)  # for reproducibility
#模型开始，用二维的模型进行处理
model = Sequential()
#BN
model.add(Convolution2D(input_shape=(32,32,1),filters=32,kernel_size=3,strides=1,padding='same',data_format='channels_last'))
#BN
model.add(Activation('relu'))

# filters, #卷积核的数目（即输出的维度）
# kernel_size,#整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
# strides=1, #整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding='valid', #补0策略，为“valid”, “same” 或“causal”，
# model.add(Convolution1D(64, 3, strides=1,border_mode='same', input_shape=(10, 32)))
# model.add(Convolution1D(32, 3, border_mode='same'))

model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_last'))
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_last'))
#BN
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
# Fully connected layer 1 input shape (64 * 5 * 5) = (1600), output shape (1024)
model.add(Flatten()) # 将响应转换为一维向量1600
model.add(Dense(1024))# 最后一层神经元个数
model.add(Activation('relu'))
model.add(Dropout(0.1)) # Dropout       DELETE
#输出的神经元个数2,2分类的问题
model.add(Dense(2))
model.add(Activation('softmax'))
adam = Adam(lr=1e-4)
#对模型进行编译，就是符合操作的一行，必须添加，设置优化器optimizer，损失函数loss
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#打印 训练中
print('Training .............................')
#模型的拟合训练测试集 训练的一些参数 几轮 ：epochs   一批次几个batch_size
#可以改的参数epochs=2, batch_size=5！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
model.fit(X_train, y_train, epochs=10, batch_size=50)

#用验证集验证：test的情况 损失函数情况和精确度情况
print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)