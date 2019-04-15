#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
import numpy as np
import math
from pandas import Series
from nltk.stem.porter import PorterStemmer  # 利用nltk的Prot算法进行词干还原，但是。。。效果不理想
# 原始文本
doc_list = [
'Julie loves me more than Linda loves me', 
'Jane likes me more than Julie loves me', 
'He likes basketball more than baseball',
]
# 此时文本
test_text = 'do you like me'
# 原始文本
# doc_list = [
# '马上就要下雨了', 
# '今天好像要下雨', 
# '我中午想吃面',
# ]
# 分词过后的数组
participle_arr = []
# 词向量最终结果
doc_term_matrix = []
# 词向量
label = []
"""
    分词
    得出label
    @param text 词向量
"""
def participle_to_label(text):
    # 词向量
    word_dict = dict()
    for doc in text:
        doc_arr = doc.split()
        origin_arr = []
        participle_arr.append(doc_arr)
    # 将结果转换为set
    word_set = set(participle_arr[0]).union(participle_arr[1]).union(participle_arr[2])
    for word in word_set:
        label.append(word)
    print("处理过后的label是%s"%label)
    # 将set转换为dict
    i = 0
    word_dict = dict()
    for word in word_set:
        word_dict[word] = i
        i+=1
    return word_dict,participle_arr

def freq(word, document): 
    return document.split().count(word)
"""
    将participle_arr数组中的数据转换为对应dict中的位置
"""
def tf(word_dict, participle_arr):       
    index = 0
    index_arr = []
    for participle_word in participle_arr:
        index_list = []
        for word in participle_word:
            index = word_dict[word]
            index_list.append(index)
        participle_word = []
        participle_word = index_list
        index_arr.append(participle_word)
    participle_arr = []
    participle_arr = index_arr
    return participle_arr
"""
    计算词向量
"""
def tf_method(word_dict, participle_arr):
    participle_arr = tf(word_dict,participle_arr)
    # 计算词向量
    for participle_word in participle_arr:
        # 单个结果
        vectorLists = []
        # 获取dict集合中所有的values和participle_arr中的数字比较
        for index in word_dict.values():
            vectorLists.append(participle_word.count(index))
        doc_term_matrix.append(vectorLists)
    # print("原始文本词向量是:%s" %(doc_term_matrix))
"""
   计算df
   单个词向量在整个句子中的数量
"""
def numDocsCounts(word):
    doccount = 0
    for doc in doc_list:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount

"""
    idf 计算
"""
def idf(word):
    n_samples = len(doc_list)
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
    处理测试句子的词向量
"""
def test_tf(word_dict, test_text):
    # 分词
    test_list = test_text.split()
    # 计算词向量
    vector_list = []
    for index in word_dict.keys():
         vector_list.append(test_list.count(index))
    # print("测试文本词向量是:%s" %(vector_list))   
    return vector_list
def test_idf(test_text):
    my_idf_vector = []
    test_idf = []
    for word in label:
        doccount = 0
        n_samples = len(test_text)
        if test_text.split().count(word) > 0:
                doccount += 1
        df = doccount
        idf = 0
        if df == 0:
            idf = 0
        else:
            idf =np.log(n_samples / df + 1)
        my_idf_vector.append(idf)
    for freq in my_idf_vector:
        test_idf.append(format(freq, 'f'))
    return test_idf
"""
    相似度计算
    @test_text 测试文本向量
    @doc_term_matrix 原始文本向量
    @num 数量
"""
def cosion(test_text,doc_term_matrix,num):
    vector1 = np.array(test_text)
    vector2 = np.array(doc_term_matrix)
    result = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    print("测试文本和第%d句话的相似度是%0.6f"%(num, result))    

"""
    计算tfidf
    其中包含了归一化处理
"""
def tfidfs(tf,idf):
    tfidfs = []
    for i in range(np.shape(tf)[0]):
        word_tfidf = np.multiply(tf[i], idf)
        tfidfs.append(word_tfidf)
    # 归一化处理
    normalizer_tfidfs = []
    for i in range(np.shape(tfidfs)[0]):
        word_tfidf = l2_normalizer(tfidfs[i])
        normalizer_tfidfs.append(word_tfidf)
    print("原始文本：",normalizer_tfidfs)
    return normalizer_tfidfs
"""
    归一化处理
"""
def  l2_normalizer(vec): 
       denom = np.sum([el**2 for el in vec]) 
       return [(el / math.sqrt(denom)) for el in vec]
if __name__ == "__main__":
    # 处理原始文本数据得到label
    word_dict, participle_arr = participle_to_label(doc_list)
    # 根据label计算词向量
    tf_method(word_dict,participle_arr)
    tf = np.array(doc_term_matrix).astype(int)
    idf = np.array(idf_method()).astype('float')
    normalizer_tfidfs = tfidfs(tf,idf)
    # 向量化测试文本集合
    test_tf_list = np.array(test_tf(word_dict,test_text)).astype(int) 
    test_idf_list = np.array(test_idf(test_text)).astype('float')
    test_tfidfs = test_tf_list * test_idf_list
     # 对测试文本的tfidfs进行归一化处理
    normalizer_test_tfidfs = []
    normalizer_test_tfidfs.append(l2_normalizer(test_tfidfs))
    print("测试文本：",normalizer_test_tfidfs)
    # 依次比较每一句话和测试文本的相似程度
    i = 1
    for doc in normalizer_tfidfs:
        cosion(normalizer_test_tfidfs, doc, i)
        i+=1