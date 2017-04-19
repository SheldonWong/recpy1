#coding=utf8
import numpy as np
import math

from utils.logger import get_logger

logger = get_logger('lmf')

class Lmf:

	

	'''
	@desc:训练
	@param:trainMatrix,k(int,代表factors数量)
	@return:u,v(m*k和n*k的矩阵)
	'''
	def train(self,a,k):
	    m = len(a)
	    n = len(a[0])
	    alpha = 0.01
	    lamda = 0.01
	    u = np.random.rand(m,k)
	    v = np.random.rand(n,k)
	    print('parameter of tarin:',m,n,np.shape(u),np.shape(v))
	    
	    for t in range(20):
	        loss = 0.0
	        for i in range(m):
	            for j in range(n):
	                if(math.fabs(a[i][j]) > 1e-4):
	                    err = a[i][j] - np.dot(u[i],v[j])
	                    for r in range(k):
	                        gu = err * v[j][r] - lamda * u [i][r]
	                        gv = err * u[i][r] - lamda * v [j][r]
	                        u[i][r] += alpha * gu
	                        v[j][r] += alpha * gv
	                    loss += err 
	        logger.info("t:%d====================loss:%f"%(t,loss))
	    return u,v

	'''
	@desc: 获取预测评分列表
	@param: u,v 
	@return: 预测评分列表
	'''
	def prediction(self,u,v,testset):
		prediction = []
		for row in testset.itertuples():
			i = row[1]
			j = row[2]
			rating = np.dot(u[i],v[j].T)
			prediction.append([i,j,rating])
		return prediction


	'''
	@desc:评测,主要评测指标MSE
	@param: u,v testset
	@return: mse
	'''
	def evaluation(self,prediction,testset):
		prediction = np.array(prediction)
		mse = np.sum( (prediction[:,2] - testset['rating'])**2 ) / len(testset)
		return mse 
