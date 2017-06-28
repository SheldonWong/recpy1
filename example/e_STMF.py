#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from model import stmf
from textblob import TextBlob 
import collections
import numpy as np 
import pandas as pd



logger = get_logger('e_STMF')


filename = 'j:/amazon/output2/Arts@uirr.csv'
dataname = filename.split('/')[-1]
#outpath
outpath = 'j:/amazon/result/result_stmf/result1/'
'''
#0.读取数据
ratings = preprocess.readdata(filename,',')
#1.判断是否有重复元素，如果有，去除重复元素
ratings_d = preprocess.drop_duplicate(ratings)
#2. 替换user_id 与 item_id
ratings_r,users,items = preprocess.replace_user_and_item(ratings_d) 


#基本数据描述(包含数据总数目，用户数，物品数)
#用户数
m = len( users )
n = len( items )
logger.info('dataset:'+dataname+',ratings:'
	+str(len(ratings_r))+',user:'+str(m)+',item:'+str(n))


#3. 切分数据
trainset,testset = preprocess.split_data(ratings_r,0.8)
trainset.to_csv(outpath+'trainset'+'_'+dataname,index=None,header=None)
testset.to_csv(outpath+'testset'+'_'+dataname,index=None,header=None)
'''
#4 获取trainset中的user和item，因为后面分别有一次根据user和item聚合


#5 计算情感值
trainfile = 'j:/amazon/result/result_stmf/result1/trainset_Arts@uirr.csv'
train_ratings = preprocess.readdata(trainfile,',')

review_series = train_ratings['review']
review_l = list(review_series)

sentiment_l = []
for review in review_l:
	t = TextBlob(review)
	sentiment_l.append(t.polarity)

#根据item聚合review
d = collections.OrderedDict()
for row in train_ratings.itertuples():
	if(row[2] not in d.keys()):
		d[row[2]] = row[4]
	else:
		d[row[2]] = d.get(row[2]) + '===' + row[4]

item_review_l = list(d.items())
print(item_review_l[0][1])

#分词，去除停用词，stem，


#构建词典


#构建词袋


#lda 输出

