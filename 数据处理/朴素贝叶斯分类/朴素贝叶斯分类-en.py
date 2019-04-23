#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import os
import numpy as np
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
    for document in train_data_set:
        posting_list.append(document[0].split())
    train_data_set = posting_list
    #对应上述6篇文章的分类结果，1为侮辱性，0为非侮辱性
    classVec = [0,1,0,1,0,1] 
    return train_data_set,classVec 

"""
    创建一个没有重复词汇得列表
"""
def createVocabList(dataSet):# 将所有文章中的词汇取并集汇总
    vocabSet = set([])  # 定义一个set(set存储的内容无重复)
    for document in dataSet:# 遍历导入的dataset数据，将所有词汇取并集存储至vocabSet中
        vocabSet = vocabSet | set(document) # | 符号为取并集，即获得所有文章的词汇表
    return list(vocabSet)


"""
    向量化训练句子
"""
def setOfWordVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)   # 生成一个全部为0得长度为向量次长度得向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    return returnVec

"""
    朴素贝叶斯分类器训练函数
    @trainMatrix    文章矩阵
    @trainCategory  训练分类
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)        #计算有多少篇文章
    numWords = len(trainMatrix[0])         #计算第一篇文档的词汇数
    pAbusive = sum(trainCategory) / float(numTrainDocs) #计算p(c1)，p(c0)=1-p(c1)
    p0Num = np.zeros(numWords)             #构建一个空矩阵，用来计算非侮辱性文章中词汇数
    p1Num = np.zeros(numWords)             #构建一个空矩阵，用来计算侮辱性文章中词汇数
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):          #遍历每一篇文章，来求P(w|c)
        if trainCategory[i] == 1:          #判断该文章是否为侮辱性文章
            p1Num += trainMatrix[i]        #累计每个词汇出现的次数
            p1Denom += sum(trainMatrix[i]) #计算所有该类别下不同词汇出现的总次数
        else:                              #如果该文章为非侮辱性文章
            p0Num += trainMatrix[i] 
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom                 #计算每个词汇出现的概率P(wi|c1)
    p0Vect = p0Num/p0Denom                 #计算每个词汇出现的概率P(wi|c0)
    return p0Vect,p1Vect,pAbusive

"""
    朴素贝叶斯分类函数
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return "带有侮辱性的词语"
    else:
        return "正常语句"

"""
    测试贝叶斯分类
"""
def testNB():
    # 加载数据集
    train_data_set,classVec  = load_data_set()
    # 创建词集合
    vocab_set = createVocabList(train_data_set)
    # 生成向量
    trainMat = []
    for postinDoc in train_data_set:
        trainMat.append(setOfWordVec(vocab_set, postinDoc))
    # 朴素贝叶斯分类器训练函数
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(classVec))

    test_entry = ['love', 'my', 'dalmation']

    thisDoc = np.array(setOfWordVec(vocab_set, test_entry))

    print(test_entry, '分类是:', classifyNB(thisDoc, p0V, p1V, pAb))

    test_entry = ['stupid', 'garbage']

    thisDoc = np.array(setOfWordVec(vocab_set, test_entry))

    print(test_entry, '分类是:', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == "__main__":
    testNB()



 