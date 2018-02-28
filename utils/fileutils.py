#coding=utf8
import sys
import gzip
import os
from logger import get_logger

'''
fileutils模块有两个作用
1. 提供常用file(包括目录和文件) API

'''

'''
@desc:读取目录，返回一个全路径文件列表
@param：path,filter
@return：pathList
note:filter是过滤器，留下哪些文件，例如只留下gzip格式的文件
理论上应该通过文件的魔数判断，
这里偷懒使用文件名称判断，只要.gz存在于文件名称中就认为是gzip格式的文件
'''
def read_dir_by_filter(root_dir,filter):
    file_list = []
    for root,dirs,files in os.walk(root_dir):
        for filepath in files:
            if(filter in filepath):
                file_list.append(os.path.join(root,filepath))
    return file_list

def read_dir(root_dir):
    file_list = []
    for root,dirs,files in os.walk(root_dir):
        for filepath in files:
            file_list.append(os.path.join(root,filepath))
    return file_list


'''
get_file_name():通过路径获取文件名，将来放到util模块中
@param:path
@return:filename
'''
def get_file_name(path):
    return path.split('/')[-1].split('.')[0]



filelist = read_dir('J:/amazon/output2')
print(filelist)