#coding=utf8
import sys
sys.path.append("..")
import heapq
import pandas as pd
import numpy as np 
import nlp.nlp_utils as nu 
from gensim import corpora, models 
from keras.models import Model #泛型模型  
from keras.layers import Dense, Input  


'''
从评论中提取商品的深度特征
'''



def getDeepFeature(trainset):
    #0. 根据商品聚合评论
    d = 3000
    item_review_df = nu.getItemReview(trainset)
    doc_clean_set = [nu.doc_clean(doc) for doc in item_review_df['reviews']]
    corpus,diction = nu.getDict(doc_clean_set)
        
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus] 

    vector_list = []
    for item in corpus_tfidf:
        vector = []
        for (k,v) in item:
            vector.append(v)
        #生成固定长度的向量
        if(len(vector) < d):
            size = d-len(vector)
            #np.random.rand(size).tolist()
            vector = vector + [0.0] * size
            #vector = vector + (0.001 * np.random.rand(size)).tolist()
        if(len(vector) > d):
            vector = heapq.nlargest(d, vector)
        vector_list.append(vector)

    print(len(vector_list))
    print(len(vector_list[0]))
    x_train = np.array(vector_list) 
    #1. TF-IDF 5000 vector_list
    print(x_train.shape)

    #2. DAE  1000 100 20 5

    # 压缩特征维度至5维  
    encoding_dim = 5  
      
    # this is our input placeholder  
    input_img = Input(shape=(d,)) 

    # 编码层
    encoded = Dense(800, activation='relu')(input_img)  
    encoded = Dense(100, activation='relu')(encoded)  
    encoded = Dense(20, activation='relu')(encoded)  
    encoder_output = Dense(encoding_dim)(encoded)

    # 解码层
    decoded = Dense(20, activation='relu')(encoder_output)  
    decoded = Dense(100, activation='relu')(decoded)  
    decoded = Dense(800, activation='relu')(decoded)  
    decoded = Dense(d, activation='tanh')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)

    encoder = Model(inputs=input_img, outputs=encoder_output) 

    autoencoder.compile(optimizer='adam', loss='mse') 

    autoencoder.fit(x_train, x_train, epochs=1, batch_size=256, shuffle=True)

    encoded_imgs = encoder.predict(x_train)



    doc_topic = encoded_imgs
    doc_topic_df = pd.DataFrame()
    doc_topic_df['item'] = list(item_review_df['item'])
    doc_topic_df['topic'] = doc_topic.tolist()

    #构建item_vector字典
    item_id_l = doc_topic_df['item']
    doc_topic_l = doc_topic_df['topic']

    item_vector_dict = dict(zip(item_id_l,doc_topic_l))
    return doc_topic_df,item_vector_dict


'''
df = pd.DataFrame(encoded_imgs)
df.to_csv('./feature.csv',index=None,header=None)

trainset = pd.read_csv('../data/trainset_Arts@uirr.csv')
getDeepFeature(trainset)
'''