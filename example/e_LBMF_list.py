#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
from preprocess import preprocess
from model import lbmf
import pandas as pd
import numpy as np 


logger = get_logger('e_LBMF_list')

filename = 'j:/amazon/output/Arts@uir.csv'
dataname = filename.split('/')[-1]
#outpath
outpath = 'j:/amazon/result/result10/'
#0.读取数据
ratings = preprocess.readdata(filename,',')
#1.判断是否有重复元素，如果有，去除重复元素
#ratings_d = preprocess.drop_duplicate(ratings)
#2. 替换user_id 与 item_id
ratings_r,users,items = preprocess.replace_user_and_item(ratings) 

users_df = pd.DataFrame(users)
items_df = pd.DataFrame(items)
users_df.to_csv(outpath+'usersList'+'_'+dataname,index=None,header=None)
items_df.to_csv(outpath+'itemsList'+'_'+dataname,index=None,header=None)

#基本数据描述(包含数据总数目，用户数，物品数)
#用户数
m = len( users )
n = len( items )
logger.info('dataset:'+dataname+',ratings:'+str(len(ratings_r))+',user:'+str(m)+',item:'+str(n))


#3. 切分数据
trainset,testset = preprocess.split_data(ratings_r,0.8)
trainset.to_csv(outpath+'trainset'+'_'+dataname,index=None,header=None)
testset.to_csv(outpath+'testset'+'_'+dataname,index=None,header=None)



#4. 构建训练输入
#train_matrix = preprocess.create_matrix_by_trainset(trainset,m,n)
train_list = preprocess.create_train_list(trainset)


'''
#类型检测
l = []
print(type(train_matrix) is np.ndarray)b
print(isinstance(train_matrix,np.ndarray))
print(type(l))
print(isinstance(l,list))
exit()
'''

#5. 训练
MF = lbmf.Lbmf(train_list,users,items,5,20,0.9,0.01)
u,v,bu,bi,global_bias = MF.train_by_list()

#5.1 模型保存
'''
u_pd = pd.DataFrame(u)
v_pd = pd.DataFrame(v)
bu_pd = pd.DataFrame(bu)
bi_pd = pd.DataFrame(bi)
u_pd.to_csv(outpath+'user_feature'+'_'+dataname,index=None,header=None)
v_pd.to_csv(outpath+'item_feature'+'_'+dataname,index=None,header=None)
bu_pd.to_csv(outpath+'bu_feature'+'_'+dataname,index=None,header=None)
bi_pd.to_csv(outpath+'bi_feature'+'_'+dataname,index=None,header=None)

'''
#6. 获取预测列表
prediction_list = []
for row in testset.itertuples():
	index = row[0]
	i = row[1]
	j = row[2]
	rating_hat = MF.predict(i,j,True)		
	prediction_list.append([index,i,j,rating_hat])
	
df = pd.DataFrame(prediction_list)
df.to_csv(outpath+'predictionList'+'_'+dataname,index=None,header=None)

#6. 预测分数分布，高的多还是低的多




#7. 评测
mse = MF.evaluation(prediction_list,testset)
logger.info("MSE:"+str(mse))
logger.info("result has been put in "+outpath)