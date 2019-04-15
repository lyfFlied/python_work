#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
import numpy as np
import data_handle as dh

cat_image_list,dog_image_list = dh.load_data()
pictureClasses=[0]*200+[1]*200

"""
    朴素贝叶斯适配器训练函数
"""
def NBFit(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.zeros(trainMatrix[0].shape); p1Num = np.zeros(trainMatrix[0].shape)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            # 矩阵加法这里有点问题
            arr =  np.array(trainMatrix[i])
            print(p0Num.shape)
            print(arr.shape)
            p0Num += arr
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    P0Vect = p0Num/p0Denom
    return P0Vect,p1Vect,pAbusive


P0Vect,p1Vect,pAbusive=NBFit(cat_image_list,pictureClasses)
print(P0Vect,p1Vect,pAbusive)