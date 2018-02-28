#coding=utf8

from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import numpy as np 
import random
import string
import collections
import pandas as pd


'''
@description:预处理，包含分词，去除停用词，标点，stem
@param:doc_set
@return:doc_set_clean
'''


'''
def doc_clean(doc_set):
    stop_words = nltk.corpus.stopwords.words('english')
    exclude = string.punctuation 
    wnl = WordNetLemmatizer()
    doc_set_clean = []
    for doc in doc_set:
    	#去除停用词
        stop_free = ' '.join([w for w in doc.lower().split() if w not in stop_words])
        #去除标点符号
        punc_free = ''.join([w for w in stop_free if w not in exclude])
        #词型归并（词干化）
        normalized = ''.join([wnl.lemmatize(w) for w in punc_free])
        #去除长度小于3的词汇
        result = ''.join()
        doc_set_clean.append(normalized)
    return doc_set_clean
'''

'''
@description:通过文档集建立不重复的词典
@param:doc_set_clean
@return:vocabSet_l
'''
def doc_vocabulary(doc_set_clean):
    vocabSet = set()
    for doc in doc_set_clean:
        vocabSet = vocabSet | set(nltk.word_tokenize(doc))
    vocabSet_l = vocabSet
    return vocabSet_l

'''
@description:将文档表示成文档向量(元素是词频)
@param:vocabSet,inputdoc
@return:
'''
def doc_bow(vocabSet,inputdoc):
	returnVec = [0 for t in range(len(vocabSet))]
	for word in inputdoc:
		if word in vocabSet:
			returnVec[vocabSet.index(word)] += 1
	return returnVec


'''
@description:将文档集表示成文档-词频矩阵
@param:vocabSet_l,doc_set_clean
@return:doc_matrix(ndarray)
'''
def doc_matrix(vocabSet_l,doc_set_clean):
    doc_matrix = []
    for doc in doc_set_clean:
        doc_matrix.append(bow(voc,doc))
    return np.array(doc_matrix)


'''
@desc:根据item聚合评论
@param:训练集
@return:item_review_df 
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
@return:corpus,diction
'''
def getDict(doc_clean_set):
    diction = corpora.Dictionary([doc.split() for doc in doc_clean_set])
    corpus = [diction.doc2bow(doc.split()) for doc in doc_clean_set]
    return corpus,diction
