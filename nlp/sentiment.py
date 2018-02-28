#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from textblob import TextBlob
import pandas as pd 
import numpy as np 


'''
构建用户偏好矩阵
1. 根据训练集中的评论，计算情感
2. 根据用户购买集和item-vector矩阵，计算用户偏好矩阵
'''


#1. 根据评论，计算用户-物品-情感列表
def get_sentiment(trainset):
    review_series = trainset['review']
    review_l = list(review_series)

    sentiment_l = []
    for row in trainset.itertuples():
    	t = TextBlob(row[4])
    	#sentiment = (t.polarity + 1) / 2.0
    	sentiment = 0.5 * t.polarity + 0.5 * (row[3] - 3) / 2.0
    	sentiment_l.append([row[1],row[2],sentiment])

    sentiment_df = pd.DataFrame(sentiment_l)
    return sentiment_df
'''
# 2.根据item_review和doc_topic,构建item_vector字典
doc_topic_df = pd.read_csv('../data/out/doc_topic.csv',header=None)
item_review_df = pd.read_csv('../data/out/item_review.csv',header=None)

item_id_l = list(item_review_df[0])
doc_topic_l = doc_topic_df.as_matrix().tolist()

item_vector_dict = dict(zip(item_id_l,doc_topic_l))
'''


#用户购买集,dict
def get_userbuy(trainset):
    user_buy_dict = {}

    for row in trainset.itertuples():
        if(row[1] not in user_buy_dict):
            t = list()
            t.append(row[2])
            user_buy_dict[row[1]] = t
        else:
            user_buy_dict.get(row[1]).append(row[2])
            #print(user_buy_dict.get(row[1]))
    print('len of user_buy_dict:{}'.format(len(user_buy_dict)))
    return user_buy_dict

# 5*m item_vector  m*1 user_item_sentiment  5*1
#遍历user_buy_dict,构造user_preference_dict

'''
uisv分别代表
u：用户
i：用户购买集
s：情感集
v：特征集
p: 偏好集(k * 1)
'''
def get_uisv(sentiment_df,user_buy_dict,item_vector_dict):

    #先构造 uis_dict,key=(u,i),value=sentiment
    uis_dict = {}
    for row in sentiment_df.itertuples():
        uis_dict[(row[1],row[2])] = row[3]
    #print('len of uis_dict:{}'.format(len(uis_dict)))
    #print(uis_dict.get((2690,154)))

    # 对每个用户，构造两个list，分别是:
    # user_item_sentiment_l m * 1
    # user_item_vector_l 5 * m

    us_dict = {}
    uv_dict = {}

    for row in user_buy_dict.items():
        user_item_sentiment_l = []
        user_item_vector_l = []
        user = row[0]

        for item in row[1]:
            user_item_sentiment_l.append(uis_dict.get((user,item)))
            user_item_vector_l.append(item_vector_dict.get(item))
        us_dict[user] = user_item_sentiment_l
        uv_dict[user] = user_item_vector_l

    #print('len of us_dict:{}'.format(len(us_dict)))
    #print('len of uv_dict:{}'.format(len(uv_dict)))

    uisv_df = pd.DataFrame()
    uisv_df['user'] = user_buy_dict.keys()
    uisv_df['item'] = user_buy_dict.values()
    uisv_df['sentiment'] = us_dict.values()
    uisv_df['vector'] = uv_dict.values()

    pref_l = []
    for row in uisv_df.itertuples():
        i_v = np.array(row[4])
        s_v = np.array(row[3])
        pref = np.dot(i_v.T,s_v.T) / len(s_v)
        pref_l.append(pref.tolist())

    uisv_df['pref'] = pref_l
    return uisv_df

