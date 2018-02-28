#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
import math
import numpy as np 
import math

'''
以ItemFeature为初始feature，训练u
dltmf:
logistic sentiment-topic based matrix factorization
sentiment: lexicon-based
topic: TFIDF + AutoEncoder

avg = (self.ave - 1.0)/4
global_bias = math.log(avg/(1-avg))
score = global_bias  + bu[i] + bi[j] + np.dot(self.u[i],k2 * self.v[j])
sig_score = 1.0 / (1+math.exp(-score))
rating_hat = min_rating + sig_score * (max_rating - min_rating)

目标函数(需要加罚项)：

'''

logger = get_logger('dltmf')

class DLTMF:

    def __init__(self,dataname,a,testset_df,m,n,u_dict,v_dict,k,t,alpha,lamda):
        self.a = a
        self.dataname = dataname
        self.testset_df = testset_df

        self.m = m
        self.n = n


        sum = 0.0
        for row in self.a:
            sum += row[1]
        self.ave = sum / len(a) 
        avg = (self.ave - 1.0) / 5.0
        self.global_bias = math.log(0.5 * avg/(1-avg))

        self.t = t
        self.k = k
        self.alpha = alpha
        self.lamda = lamda
        self.min_rating = 1.0
        self.max_rating = 5.0

        self.bu = np.random.rand(self.m)
        self.bi = np.random.rand(self.n)


        self.u = u_dict
        self.v = v_dict

    
    def before_train(self):
        logger.info('dataset:%s'%(self.dataname))

    def train_by_list(self):
        self.before_train()
        for t in range(self.t):
            loss = 0.0
            for row in self.a:
                i = row[0][0]
                j = row[0][1]

                p = np.array(self.u.get(i))
                q = np.array(self.v.get(j))

               
                score = self.global_bias  + self.bu[i] + self.bi[j] + np.dot(p,q)

                sig_score = 1.0 / (1+math.exp(-score))
                rating_hat = self.min_rating + sig_score * (self.max_rating - self.min_rating)
                err = row[1] - rating_hat
                for r in range(self.k):
                    gu = err * sig_score * sig_score * (1 - sig_score) * self.v[j][r] - self.lamda * self.u[i][r]
                    gv = err * sig_score * sig_score * (1 - sig_score)  * self.u[i][r] - self.lamda * self.v[j][r] 
                    self.u[i][r] += self.alpha * gu
                    self.v[j][r] += self.alpha * gv
                    #print(i,j,p,q,gu,gv)
                
                self.bu[i] += self.alpha * (err * sig_score * sig_score * (1 - sig_score) - self.lamda * self.bu[i])
                self.bi[j] += self.alpha * (err * sig_score * sig_score * (1 - sig_score) - self.lamda * self.bi[j])
                
                loss += err ** 2
            prediction_list = self.prediction(self.testset_df)
            mse = self.evaluation(prediction_list)
            logger.info("t:%d====================loss:%f,mse:%f"%(t,loss,mse))

            if(math.fabs(loss) < 150):
                break
            if(t % 5 == 0):
                self.alpha = 0.99 * self.alpha
        return self.u,self.v,self.bu,self.bi,self.global_bias 

    def predict(self,i,j,bound=False):
        if(i in self.u.keys()):
            p = np.array(self.u.get(i))
        else:
            p = np.random.rand(1,self.k)
        if(j in self.v.keys()):
            q = np.array(self.v.get(j))
        else:
            q = np.random.rand(self.k,1)
        if(i not in self.u.keys() and j not in self.v.keys()):
            rating_hat = self.ave
            return rating_hat
        
        score = self.global_bias  + self.bu[i] + self.bi[j] + np.dot(p,q)
        sig_score = 1.0 / (1.0 +math.exp(-score))
        rating_hat = self.min_rating + sig_score * (self.max_rating - self.min_rating)
        
        if(bound):
            if (rating_hat > 5):
                return 4.8
            if (rating_hat < 1):
                return 1.2
        return rating_hat

    '''
    @desc: 获取预测评分列表
    @param: u,v 
    @return: 预测评分列表
    '''
    def prediction(self,testset):
        prediction = []
        for row in testset.itertuples():
            i = row[1]
            j = row[2]
            r = row[3]
            rating_hat = self.predict(i,j,True)
            prediction.append([i,j,r,rating_hat])
        return prediction

    '''
    @desc:评测,主要评测指标MSE
    @param: u,v testset
    @return: mse
    '''
    def evaluation(self,prediction):
        prediction = np.array(prediction)
        mse = np.sum( (prediction[:,3] - prediction[:,2]) ** 2 ) / len(prediction)
        return mse 

