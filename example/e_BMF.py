#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from model import bmf
import pandas as pd
import numpy as np 


logger = get_logger('e_BMF')

filename = 'j:/amazon/output/Gift_Cards@uir.csv'
dataname = filename.split('/')[-1]
#0.读取数据
ratings = preprocess.readdata(filename,',')
#1.判断是否有重复元素，如果有，去除重复元素
ratings_d = preprocess.drop_duplicate(ratings)
#2. 替换user_id 与 item_id
ratings_r = preprocess.replace_user_and_item(ratings_d) 


#基本数据描述(包含数据总数目，用户数，物品数)
#用户数
m = len( set(ratings_r['user_id']) )
n = len( set(ratings_r['item_id']) )
logger.info('dataset:'+dataname+',ratings:'+str(len(ratings_r))+',user:'+str(m)+',item:'+str(n))


#3. 切分数据
trainset,testset = preprocess.split_data(ratings_r,0.8)
trainset.to_csv('j:/amazon/train/trainset.csv',index=None,header=None)
testset.to_csv('j:/amazon/train/testset.csv',index=None,header=None)

mu = np.sum(trainset['rating']) / len(trainset)
#4. 构建训练矩阵
train_matrix = preprocess.create_matrix_by_trainset(trainset,m,n)

#5. 训练
MF = bmf.Bmf(mu)
u,v,bu,bi = MF.train(train_matrix,5)


#6. 获取预测列表
prediction_list = MF.prediction(u,v,bu,bi,testset)
df = pd.DataFrame(prediction_list)
df.to_csv('predictionList.csv',index=None,header=None)


#7. 评测
mse = MF.evaluation(prediction_list,testset)
logger.info("MSE:"+str(mse))