#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from textblob import TextBlob 
from model import stmf
import pandas as pd
import numpy as np 


logger = get_logger('e_STMF')

filename = 'j:/amazon/output2/Arts@uirr.csv'
dataname = filename.split('/')[-1]
#outpath
outpath = 'j:/amazon/result/result_stmf/result1'
#0.读取数据
ratings = preprocess.readdata(filename,',')


'''
for row in ratings.itertuples():
	print(type(row))
	print(row)
	print(row[0])
	break
==============output=============
<class 'pandas.core.frame.Pandas'>
Pandas(Index=0, user_id='A1QA985ULVCQOB', item_id='B000GKXY4S', rating=5.0, review='I really enjoy these scissors for my inspiration books that I am making (like collage, but in books) and using these different textures these give is just wonderful, makes a great statement with the pictures and sayings. Want more, perfect for any need you have even for gifts as well. Pretty cool!')
0
'''

review_series = ratings['review']
review_l = list(review_series)

sentiment = []
for review in review_l:
	t = TextBlob(review)
	sentiment.append(t.polarity)
print(sentiment[:20])