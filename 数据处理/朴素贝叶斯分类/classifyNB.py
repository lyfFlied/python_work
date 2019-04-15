#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
import numpy as np
import data_handle as dh

cat_image_list,dog_image_list = dh.load_data()
pictureClasses=[0]*200+[1]*200

def NBFit(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    P0Vect = p0Num/p0Denom
    return P0Vect,p1Vect,pAbusive


pFVect,pUVect,pCat=NBFit(cat_image_list,pictureClasses)
print(pFVect,pUVect,pCat)