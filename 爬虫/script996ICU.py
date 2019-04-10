#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

# 996ICU 爬虫

import urllib.request
from bs4 import BeautifulSoup

# 项目地址
url_path = "https://github.com/Y1ran/996.Law/issues"
# 基础地址
base_path = "https://github.com"

dicts = {}
"""
    获取所有数据
    title，url
"""
def __get_data(index = 1):
    
    # 处理URL
    url = url_path + "?page=" + str(index)  
    content = urllib.request.urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(content, 'lxml')
    datas = soup.find_all('a', class_='link-gray-dark v-align-middle no-underline h4 js-navigation-open')

    # 过滤数据
    for data in datas:
        name = data.string
        href = base_path + data['href']
        dicts[name] = href
        return 


def get_all_data():
    for i in range(20):
        print(i + 1)
        data = __get_data(i + 1)
        if data == {}:
            print('project，end！')
            break
        else:
            __get_data(i)
            print(dicts)

print(get_all_data())