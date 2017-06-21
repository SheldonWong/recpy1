#coding=utf8
import pandas as pd 
import numpy as np 
import random
import math
from scipy import sparse
from utils.logger import get_logger


'''
数据预处理,主要目标是创建训练要用的矩阵
包含
readdata
drop_dulpicate
replace_user_and_item
split_data
create_matrix_by_trainset
'''

'''
读取文件，返回DataFrame格式的数据
@param:filename
@return:filename_dataframe
'''

def readdata(filename,sep):
    rnames = ['user_id','item_id','rating']
    ratings = pd.read_table(filename,sep=sep,header=None,names=rnames)
    return ratings


'''
数据预处理：读取原始数据文件，去除重复元素后，选择是否输出到指定路径
@param:ratings,sep
@return:filename_drop_dulplicate.csv
'''
def drop_duplicate(ratings,isSave=False,outpath=None):
    #去除重复元素，
    #todo：重复元素保留一个，unknown的不保留，没有意义
    isDulplicatedList = ratings[['user_id','item_id']].duplicated()
    dulplicateIndexList = []
    for i in range(len(isDulplicatedList)):
        if(isDulplicatedList.iloc[i]):
            dulplicateIndexList.append(i)
    ratings = ratings.drop(dulplicateIndexList)
    if(isSave):
        ratings.to_csv(outpath+getFileName(filename)+'_drop_dulplicate'+'.csv')
        logger.info('write drop_duplicate file: '+outpath+getFileName(filename)+'_drop_dulplicate'+'.csv')
    return ratings

'''
数据预处理：传进来一个文件列表，依次做dropDulplicate处理
@param:filelist
@return:
'''
def drop_duplicate_by_filelist(filelist,sep,isSave=False,outpath=None):
    for file in filelist:
        dropDuplicate(getFileName(file),sep,isSave=isSave,outpath=outpath)

'''
@desc:考虑到文本类型的userID与ItemID
将user_id于item_id替换成相应的整型数字，其中整型数字是user_id与item_id的索引
转换前形如：A1QA985ULVCQOB, B000GKXY4S, 5.0
转换后形如：21515, 1884, 5.0
@param:ratings(DataFrame)转换前
@return:ratings(DataFrame)转换后
'''
def replace_user_and_item(ratings):
    users = list(set(ratings['user_id']))
    items = list(set(ratings['item_id']))
    l = []
    for row in ratings.itertuples():
        t = (users.index(row[1]),items.index(row[2]),row[3])
        l.append(t)
    df = pd.DataFrame(l)
    df.columns = ['user_id', 'item_id', 'rating']
    #这里有没有必要？  
    #df = df.sort_values(by=['user_id','item_id'])
    return df,users,items 
'''
根据users，items，testset还原user_id,item_id
@param:users,items,prediction_df
'''
def recover_user_and_item(users,items,prediction_df):
    l = []
    for row in prediction_df.itertuples():
        i = row[1]
        j = row[2]
        t = (users[i],items[j],row[3],row[4])
        l.append(t)
    recover_prediction_df = pd.DataFrame(l)
    return recover_prediction_df



'''
切分数据：利用DataFrame的抽样sample()方法
@param：ratings,percent(trainset/all)
@return:trainset,testset
'''
def split_data(ratings,percent):
    #一种更快捷的方式，直接利用DataFrame的抽样sample()方法
    testset = ratings.sample(frac=1-percent)
    trainset = ratings.drop(testset.index)
    return trainset,testset

'''
@desc:根据trainset构建矩阵
@param:trainset(DataFrame),m,n
@return:matrix(np.ndarray)
'''
def create_matrix_by_trainset(trainset,m,n):
    #构造矩阵
    row = trainset['user_id']
    col = trainset['item_id']
    data = trainset['rating']
    c = sparse.coo_matrix( (data,(row,col)),shape=(m,n))
    matrix = c.toarray()
    return matrix

'''
@desc:根据trainset构建一个形如[[(x,y),v]]的列表，其中(x,y)是坐标，代表user_id,item_id。v是评分。
@param:trainset
@return:train_list(python内置list)
'''
def create_train_list(trainset):
    l = []
    #row[0]=index,row[1]=user_id,row[2]=item_id,row[3]=rating
    for row in trainset.itertuples():
        t = [(row[1],row[2]),row[3]]
        l.append(t)	
    return l 

'''
@desc:根据trainset构建一个np.ndarray类型的二维数组，形如[[x,y,v]]
@param:trainset
@return:train_nd
'''
def create_train_array(trainset):
    return trainset.as_matrix()
