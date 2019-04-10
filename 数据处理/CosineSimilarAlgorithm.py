#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
import numpy as np
import math
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
        # for doc_word in doc_arr:
        #     # 利用porter进行词干还原
        #     porter_stemmer = PorterStemmer()  
        #     origin_word = porter_stemmer.stem(doc_word)
        #     origin_arr.append(origin_word)
        participle_arr.append(doc_arr)
    # 将结果转换为set
    word_set = set(participle_arr[0]).union(participle_arr[1]).union(participle_arr[2])
    # 将set转换为dict
    i = 0
    word_dict = dict()
    for word in word_set:
        word_dict[word] = i
        i+=1
    return word_dict,participle_arr

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
    print("原始文本词向量是:%s" %(doc_term_matrix))

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
    print("测试文本词向量是:%s" %(vector_list))   
    return vector_list


 # 计算余弦相似度
def cosion(test_text,doc_term_matrix,num):
    sum = 0
    sq1 = 0
    sq2 = 0
    # for i in range(len(doc_term_matrix)):
    #     sum += test_text[i] * doc_term_matrix[i]
    #     sq1 += pow(test_text[i], 2)
    #     sq2 += pow(doc_term_matrix[i], 2)
    # try:
    #     result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    # except ZeroDivisionError:
    #     result = 0.0
    # print("测试文本和第%d句话的相似度是%0.2f"%(num, result))
    vector1 = np.array(test_text)
    vector2 = np.array(doc_term_matrix)
    op7=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    print("测试文本和第%d句话的相似度是%0.6f"%(num, op7))    

if __name__ == "__main__":
    # 处理原始文本数据得到label
    word_dict, participle_arr = participle_to_label(doc_list)
    # 根据label计算词向量
    tf_method(word_dict,participle_arr)
    # 向量化测试文本集合
    test_vector_list = test_tf(word_dict,test_text)
    i = 1
    for doc in doc_term_matrix:
        cosion(test_vector_list,doc,i)
        i+=1