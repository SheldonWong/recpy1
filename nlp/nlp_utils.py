#coding=utf8

from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np 
import random
import string


'''
@description:预处理，包含分词，去除停用词，标点，stem
@param:doc_set
@return:doc_set_clean
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
        
        doc_set_clean.append(normalized)
    return doc_set_clean

'''
@description:通过文档集建立不重复的词典
@param:doc_set_clean
@return:vocabSet_l
'''
def doc_vocabulary(doc_set_clean):
    vocabSet = set()
    for doc in doc_set_clean:
        vocabSet = vocabSet | set(nltk.word_tokenize(doc))
    vocabSet_l = list(vocabSet)
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
        doc_matrix.append(doc_bow(vocabSet_l,doc))
    return np.array(doc_matrix)






