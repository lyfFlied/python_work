#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import os
import numpy as np

# #获取当前路径
path = os.getcwd() + "\\数据处理\\朴素贝叶斯分类\\sentence.txt"

# # 数据预处理
# """
#     读取数据
# """
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

if __name__ == "__main__":
    # 创建一个向量集合存放处理好的测试集合的文本向量
    trainMat = []
    train_data_set,classVec  = load_data_set()
    # 创建词集合
    vocab_set = createVocabList(train_data_set)
    for postinDoc in train_data_set:
        trainMat.append(setOfWordVec(vocab_set, postinDoc))
    print(trainMat)
    # 处理测试数据
    test_entry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWordVec(vocab_set, test_entry)).reshape(1,-1)
    print("测试数据%s" % thisDoc)
    # 导入sklearn包 开始进行计算
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    y_pred = clf.fit(trainMat, classVec).predict(thisDoc)
    """
        0----->不带有侮辱性的
        1-----> 带有侮辱性的
    """
    print("预测结果%s" % y_pred) 
    print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (len(trainMat),(classVec != y_pred).sum()))