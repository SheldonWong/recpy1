#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from model import bmf
import pandas as pd
import numpy as np 



filename = 'j:/amazon/output/Arts@uir.csv'
dataname = filename.split('/')[-1]
#outpath
outpath = 'j:/amazon/result/result1/'
#0.读取数据
ratings = preprocess.readdata(filename,',')
#1.判断是否有重复元素，如果有，去除重复元素
ratings_d = preprocess.drop_duplicate(ratings)
#2. 替换user_id 与 item_id
ratings_r,users,items = preprocess.replace_user_and_item(ratings_d) 


#基本数据描述(包含数据总数目，用户数，物品数)
#用户数


#3. 切分数据
trainset,testset = preprocess.split_data(ratings_r,0.8)

#4. 构造训练输入
#直接用list
t1 = 
