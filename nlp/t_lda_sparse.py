import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
import nlp.nlp_utils as nu
import pandas as pd 
import collections
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

from gensim import corpora, models, similarities 
import gensim
import numpy as np 

# 传进来trainset
# 1. 先根据item聚合评论 =》item review 
# 2. 清洗数据，去除标点符号，去除停用词，词形归并 =》稀疏矩阵
# 3. 稀疏矩阵转稠密矩阵
# 4. lda

'''
@desc:根据item聚合评论
@param:
@return:
'''
def getItemReview(trainset):
    d = collections.OrderedDict()
    for row in trainset.itertuples():
        if(row[2] not in d.keys()):
            d[row[2]] = row[4]
        else:
            d[row[2]] = d.get(row[2]) + ',' + row[4]

    item_review_l = list(d.values())  
    item_review_df = pd.DataFrame()
    item_review_df['item'] = d.keys()
    item_review_df['reviews'] = item_review_l 

    return item_review_df 

'''
@desc:清洗一个item对应的评论
@param:doc（一个item对应的评论）
@return:清洗后的doc
'''
def doc_clean(doc):
    stop = set(stopwords.words("english"))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized =  ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

'''
@desc:提取字典
@param:doc_clean_set(清洗后的评论数据集)
@return:corpus
'''
def getDict(doc_clean_set):
    diction = corpora.Dictionary([doc.split() for doc in doc_clean_set])
    corpus = [diction.doc2bow(doc.split()) for doc in doc_clean_set]
    return corpus,diction

'''
稀疏矩阵转稠密矩阵
稀疏矩阵一般是用[(index,value)...]存储，该函数将稀疏矩阵还原为原矩阵的样子
'''
def sparse2dense(origin,dim):
    dense = []
    for row in origin:
        l = [0] * dim
        for t in row:
            l[t[0]] = t[1]
        dense.append(l)
    return np.array(dense)


def getItemTopic(trainset):

    item_review_df = getItemReview(trainset)
    doc_clean_set = [doc_clean(doc) for doc in item_review_df['reviews']]
    corpus,diction = getDict(doc_clean_set)
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]  #得到每个文本的tfidf向量，稀疏矩阵  
    lda = models.LdaModel(corpus_tfidf, id2word = diction, num_topics = 5)

    corpus_lda = lda[corpus_tfidf]

    doc_topic = sparse2dense(corpus_lda,5)
    doc_topic_df = pd.DataFrame()
    doc_topic_df['item'] = list(item_review_df['item'])
    doc_topic_df['topic'] = doc_topic.tolist()

    #构建item_vector字典
    item_id_l = doc_topic_df['item']
    doc_topic_l = doc_topic_df['topic']

    item_vector_dict = dict(zip(item_id_l,doc_topic_l))
    return doc_topic_df,item_vector_dict

'''
测试
filename = 'j:/amazon/output2/Furniture@uirr.csv'
ratings = preprocess.readdata(filename,',')
item_vector_dict = getItemTopic(ratings)
print(len(item_vector_dict))
'''