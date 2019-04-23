#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import jieba
import os
import numpy as np
# 获取当前路径
path = os.getcwd() + "\\数据处理\\朴素贝叶斯分类\\sentence-cn.txt"
# 加载自定义词典(便于识别CXK)
jieba.load_userdict(os.getcwd() + "\\数据处理\\朴素贝叶斯分类\\newdict.txt")



"""
    读取数据
"""
def load_data_set():
    # 训练集合
    train_data_set = []
    posting_list = []
    for line in  open(path,"r",encoding="utf-8"):
        one_list = []
        one_list.append(line)
        train_data_set.append(one_list)
    for document in train_data_set:
        lists = jieba.cut(document[0], cut_all=False)
        strs = " ".join(lists)
        posting_list.append(strs.split())
    train_data_set = posting_list
    #对应上述6篇文章的分类结果，1为侮辱性，0为非侮辱性
    classVec = [1,1,1,0,0,0] 
    return train_data_set,classVec 


"""
    创建一个没有重复词汇的列表
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
    # print("转换成功的结果%s" % trainMat)
    # 处理测试数据
    test_sentence = "小心蔡徐坤娶了你"
    print("你输入的内容是:%s"%test_sentence)
    lists = jieba.cut(test_sentence, cut_all=False)
    strs = " ".join(lists)
    test_entry = strs.split()
    # 解决维度问题抛出的错误(这里需要对数据reshape一下)
    thisDoc = np.array(setOfWordVec(vocab_set, test_entry)).reshape(1,-1)
    print("测试数据%s" % thisDoc)
    # 导入sklearn包 开始进行计算
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    y_pred = clf.fit(trainMat, classVec).predict(thisDoc)
    print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (len(trainMat),(classVec != y_pred).sum()))
    """
        0----->不带有侮辱性的
        1-----> 带有侮辱性的
    """
    print("预测结果 %s " % y_pred) 
    if y_pred == 1:
        print("你在骂我!,请好好说话!")
    else:
        print("请继续!")
    