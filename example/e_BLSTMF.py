#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from textblob import TextBlob 
from model import blstmf
import collections
import numpy as np 
import pandas as pd



logger = get_logger('e_BLSTMF')


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

'''
#5 计算情感值
trainfile = 'j:/amazon/result/result_stmf/result1/trainset_Arts@uirr.csv'
train_ratings = preprocess.readdata(trainfile,',')

review_series = train_ratings['review']
review_l = list(review_series)

sentiment_l = []
for row in train_ratings.itertuples():
	t = TextBlob(row[4])
	sentiment_l.append([row[1],row[2],t.polarity])

sentiment_df = pd.DataFrame(sentiment_l)
sentiment_df.to_csv('../data/out/sentiment.csv',index=None,header=None)

'''

'''
#根据item聚合review
d = collections.OrderedDict()
for row in train_ratings.itertuples():
	if(row[2] not in d.keys()):
		d[row[2]] = row[4]
	else:
		d[row[2]] = d.get(row[2]) + '===' + row[4]

item_review_l = list(d.items())
print(item_review_l[0][1])
'''
#分词，去除停用词，stem，


#构建词典


#构建词袋


#lda 输出


trainfile = '../data/trainset_Arts@uirr.csv'
train_df = preprocess.readdata(trainfile,',')
train_list = preprocess.create_train_list(train_df)

testset_df = pd.read_csv('../data/testset_Arts@uirr.csv',header=None)

#构建两个字典，分别是用户-情感向量，物品-topic向量

#1. 构建字典user-pref
uisv_names = ['user','item','sentiment','vector','pref']
uisv_df = pd.read_csv('../data/out3/uisv3.csv',header=None,names=uisv_names)
user_l = uisv_df['user'].tolist()
pref_l = uisv_df['pref'].tolist()

pref_l_eval = []
for row in pref_l:
	pref_l_eval.append(eval(row))


u_dict = dict(zip(user_l,pref_l_eval))


#2 构建字典item-vertor
doc_topic_df = pd.read_csv('../data/out/doc_topic.csv',header=None)
item_review_df = pd.read_csv('../data/out/item_review.csv',header=None)

item_id_l = list(item_review_df[0])
doc_topic_l = doc_topic_df.as_matrix().tolist()

v_dict = dict(zip(item_id_l,doc_topic_l))




# 5. 训练
BLSTMF = blstmf.BLSTMF(train_list,testset_df,u_dict,v_dict,5,30,0.6,0.05)
wu,vv,v,bu,bi,ave= BLSTMF.train_by_list()




'''
#6. 获取预测列表
prediction_list = STMF.prediction(testset_df)
prediction_df = pd.DataFrame(prediction_list)
prediction_df.to_csv('../data/out/result.csv',index=None,header=None)
#recover_prediction_df = preprocess.recover_user_and_item(users,items,prediction_df)
#recover_prediction_df.to_csv(outpath+'predictionList'+'_'+dataname,index=None,header=None)


#7. 评测

mse = STMF.evaluation(prediction_list)
logger.info("latent offset"+str())
logger.info("MSE:"+str(mse))
#logger.info("result has been put in "+outpath)
'''