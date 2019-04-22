#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB

#获取当前路径
path = os.getcwd() + "\\数据处理\\朴素贝叶斯分类\\sentence.txt"

# 数据预处理
"""
    读取数据
"""
def load_data_set():
    # 训练集合
    train_data_set = []
    posting_list = []
    for line in  open(path):
        one_list = []
        one_list.append(line)
        train_data_set.append(one_list)
    # 分词
    for document in train_data_set:
        posting_list.append(document[0].split())
    train_data_set = posting_list
    #对应上述6篇文章的分类结果，1为侮辱性，0为非侮辱性
    classVec = [0,1,0,1,0,1] 
    return train_data_set,classVec 

train_data_set,classVec  = load_data_set()

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf = clf.fit(train_data_set, classVec)
# y_pred=clf.predict(train_data_set)
# print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (train_data_set,(classVec != y_pred).sum()))