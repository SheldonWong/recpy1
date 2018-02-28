#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from textblob import TextBlob 
from model import lstmf
import collections
import numpy as np 
import pandas as pd
from nlp import t_lda_dense as dlda


logger = get_logger('e_BLSTMF')

#输入路径（数据）
filename = 'j:/amazon/output2/Arts@uirr.csv'
dataname = filename.split('/')[-1]

#输出路径
outpath = 'j:/amazon/result/result_stmf/result1/'

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

#3. 切分数据
trainset,testset = preprocess.split_data(ratings_r,0.8)
trainset.to_csv(outpath+'trainset'+'_'+dataname,index=None,header=None)
testset.to_csv(outpath+'testset'+'_'+dataname,index=None,header=None)

#4 计算情感值(把这个踢出去)
review_series = trainset['review']
review_l = list(review_series)

sentiment_l = []
for row in train_ratings.itertuples():
    t = TextBlob(row[4])
    sentiment_l.append([row[1],row[2],t.polarity])

sentiment_df = pd.DataFrame(sentiment_l)

#5 计算主题
doc_topic = lda.getItemTopic(trainset)

#6 训练
train_list = preprocess.create_train_list(train_df)
BLSTMF = blstmf.BLSTMF(train_list,testset_df,m,n,u_dict,v_dict,5,30,0.6,0.05)
wu,vv,v,bu,bi,ave= BLSTMF.train_by_list()


#7 获取预测列表



