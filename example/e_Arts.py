#coding=utf8
import sys
sys.path.append("..")
import os
from utils.logger import get_logger
from preprocess import preprocess
from textblob import TextBlob 
from model import lstmf
import collections
import numpy as np 
import pandas as pd
from nlp import t_lda_sparse as slda
from nlp import sentiment as s


'''
@desc:Arts数据集上的实验
读取数据
预处理
切分数据
先计算主题，再计算v_dict
先计算情感值，再计算u_dict
训练
'''


#输入路径（数据）
filename = 'j:/amazon/output2/Watches@uirr.csv'
dataname = filename.split('/')[-1]

outpath = 'j:/amazon/output3/'+dataname

#0.读取数据
ratings = preprocess.readdata(filename,',')
print(len(ratings))
#1.判断是否有重复元素，如果有，去除重复元素
#ratings_d = preprocess.drop_duplicate(ratings)
#2. 替换user_id 与 item_id
ratings_r,users,items = preprocess.replace_user_and_item(ratings) 

m = len(users)
n = len(items)
#3. 切分数据
trainset,testset = preprocess.split_data(ratings_r,0.8)
print(len(trainset))
print(len(testset))

#4. 计算主题
print("step1:计算主题")
doc_topic_df,item_vector_dict = slda.getItemTopic(trainset)


#5. 计算情感
print("step2:计算情感")
sentiment_df = s.get_sentiment(trainset)
user_buy_dict = s.get_userbuy(trainset)
uisv_df = s.get_uisv(sentiment_df,user_buy_dict,item_vector_dict)

#如果目录文件不存在，则创建
if not os.path.isdir(outpath+'/out1/'):
    os.makedirs(outpath+'/out1/')
doc_topic_df.to_csv(outpath+'/out1/'+'item_topic.csv')
trainset.to_csv(outpath+'/out1/'+'trainset.csv')
testset.to_csv(outpath+'/out1/'+'testset.csv')
uisv_df.to_csv(outpath+'/out1/'+'uisv.csv')


'''
trainset = pd.read_csv('j:/amazon/output3/Jewelry@uirr.csv/out1/trainset.csv')
testset = pd.read_csv('j:/amazon/output3/Jewelry@uirr.csv/out1/testset.csv')
uisv_df = pd.read_csv('j:/amazon/output3/Jewelry@uirr.csv/out1/uisv.csv')
'''

#6. 构建字典user-pref
user_l = uisv_df['user'].tolist()
pref_l = uisv_df['pref'].tolist()

'''
pref_l_eval = []
for row in pref_l:
    pref_l_eval.append(eval(row))
'''

u_dict = dict(zip(user_l,pref_l))

#7 构建字典item-vertor
v_dict = item_vector_dict

# 8. 训练 k t alpha lamda
train_list = preprocess.create_train_list(trainset)

LSTMF = lstmf.LSTMF(dataname,train_list,testset,m,n,u_dict,v_dict,5,20,0.3,0.03)
u,v,bu,bi,ave= LSTMF.train_by_list()
