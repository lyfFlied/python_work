#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
import numpy as np
import math
import wordcloud
# 词向量
label = ["me", "basketball", "Julie", "baseball", "likes", "loves", "Jane", "Linda", "He", "than", "more"]
# 文本
docList = [
'Julie loves me more than Linda loves me', 
'Jane likes me more than Julie loves me', 
'He likes basketball more than baseball',
# 'l likes the dog',
# 'you likes you ddd'
]
"""
   @param word 特征词
   @param document 句子
"""
def freq(word, document): 
    return document.split().count(word)
def tf(word, document):
    return freq(word, document)
"""
    tf计算方法
"""
def tf_method():
    # 单个结果
    vectorLists = []
    # 最终结果
    doc_term_matrix = []
    for i in range(len(docList)):
        tf_vector_string = ''
        vectorLists = []
        for j in range(len(label)):
            tf_vector = tf(label[j], docList[i])
            vectorLists.append(int(tf_vector))
        doc_term_matrix.append(vectorLists)
        tf_vector_string = ','.join(format(freq, 'd') for freq in vectorLists)
        print("句子 %s 的词频向量是：%s" % (docList[i], vectorLists))
    return doc_term_matrix
"""
   计算df
   单个词向量在整个句子中的数量
"""
def numDocsCounts(word):
    doccount = 0
    for doc in docList:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount
"""
    idf 计算
"""
def idf(word):
    n_samples = len(docList)
    df = numDocsCounts(word)
    return np.log(n_samples / df + 1)
"""
    idf方法
"""
def idf_method():
    idf_list = []
    # 生成一个label的迭代器
    my_idf_vector  = [idf(word) for word in label]
    for freq in my_idf_vector:
        idf_list.append(format(freq, 'f'))
    idf_string = ', '.join(format(freq, 'f') for freq in my_idf_vector)
    # print(' [' + idf_string + ']')
    return idf_list

"""
    归一化处理
"""
def  l2_normalizer(vec): 
       denom = np.sum([el**2 for el in vec]) 
       return [(el / math.sqrt(denom)) for el in vec]
"""
    生成词云
"""
def get_word_cloud():


if __name__ == "__main__":
    tf = np.array(tf_method()).astype(int)
    idf = np.array(idf_method()).astype('float')
    print("===========================")
    print("tf计算结果：%s" % (tf))
    print("===========================")
    print("idf计算结果：%s" % (idf))
    print("===========================")
    tfidfs = []
    for i in range(np.shape(tf)[0]):
        print("第%d个文档" % (i))
        print("===========================")
        print(tf[i])
        print(idf)
        word_tfidf = np.multiply(tf[i], idf)
        tfidfs.append(word_tfidf)
        print(word_tfidf)
        print("===========================")
    print('总的tfidfs是:\n', np.array(tfidfs))
    # 归一化处理
    normalizer_tfidfs = []
    for i in range(np.shape(tfidfs)[0]):
        print("第%d个文档" % (i))
        print("===========================")
        word_tfidf = l2_normalizer(tfidfs[i])
        normalizer_tfidfs.append(word_tfidf)
        print(np.array(normalizer_tfidfs))