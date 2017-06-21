#coding=utf8

import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
import pandas as pd 

'''
1. 读入数据
2. 将评论按照item划分,dict
3. 预处理，去除停止词，stem
4. doc2bow
5. lda
6. item_topic
'''

filename = 'j:/amazon/output2/Arts@uirr.csv'

rnames = ['user_id','item_id','rating','review']
ratings_df = pd.read_csv(filename,header=0,names=rnames)
print(ratings_df[:5])














