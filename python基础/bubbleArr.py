#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
import datetime
def result(datas):
    for i in range(len(datas)):
        for j in range(len(datas) - 1 - i):
            if(datas[j]>datas[j+1]):
                  temp=datas[j]
                  datas[j]=datas[j+1]
                  datas[j+1]=temp           
    print("运算结果：" + str(datas))
    

if __name__ == "__main__":
    list1 = [10,5,7,60,2,1,3,12]
    print("初始数组：" + str(list1))
    start = datetime.datetime.now()
    result(list1)
    end = datetime.datetime.now()   
    print("运算耗时：" + str(end-start))