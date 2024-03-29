#coding=utf8
import sys
sys.path.append("..")
from utils.logger import get_logger
import math
import numpy as np 

'''
stmf:
sentiment-topic based matrix factorization
sentiment: naive-bayes
topic: LDA
rating_hat = mu + bu[i] + bi[j] + np.dot(wu[i]*u,wv[j]*v)
'''

logger = get_logger('stmf')

class STMF:

    def __init__(self,dataname,a,testset_df,m,n,u_dict,v_dict,k,t,alpha,lamda):
        self.dataname = dataname
        self.a = a
        self.testset_df = testset_df

        self.m = m
        self.n = n


        sum = 0.0
        for row in self.a:
            sum += row[1]

        self.ave = sum / len(a)        
        self.t = t
        self.k = k
        self.alpha = alpha
        self.lamda = lamda

        self.bu = np.random.rand(self.m)
        self.bi = np.random.rand(self.n)
        self.wu = np.random.rand(self.m)
        self.wv = np.random.rand(self.n)
        self.u = u_dict
        self.v = v_dict

    
    def before_train(self):
        pass

    def train_by_list(self):
        self.before_train()
        for t in range(self.t):
            loss = 0.0
            for row in self.a:
                i = row[0][0]
                j = row[0][1]
                predict_score = self.predict(i,j,False)
                err = row[1] - predict_score

                for r in range(self.k):
                    p = np.array(self.u.get(i))
                    q = np.array(self.v.get(j))
                    gu = err * self.wv[j] * np.dot(p,q) - self.lamda * self.wu[i]
                    gv = err * self.wu[i] * np.dot(p,q) - self.lamda * self.wv[j]
                    self.wu[i] += self.alpha * gu
                    self.wv[j] += self.alpha * gv
                    #print(i,j,p,q,gu,gv)
                
                self.bu[i] += self.alpha * (err - self.lamda * self.bu[i])
                self.bi[j] += self.alpha * (err - self.lamda * self.bi[j])
                
                loss += err ** 2
            prediction_list = self.prediction(self.testset_df)
            mse = self.evaluation(prediction_list)
            logger.info("t:%d====================loss:%f,mse:%f"%(t,loss,mse))

            if(math.fabs(loss) < 150):
                break
                        #更新学习速率
            if(t % 5 == 0):
                self.alpha = 0.9 * self.alpha
        return self.wu,self.wv,self.bu,self.bi,self.ave 

    def predict(self,i,j,bound=False):
        if(i in self.u.keys()):
            p = self.wu[i] * np.array(self.u.get(i))
        else:
            p = np.random.rand(1,self.k)
        if(j in self.v.keys()):
            q = self.wv[j] * np.array(self.v.get(j))
        else:
            q = np.random.rand(self.k,1)
        rating_hat = self.ave  + self.bu[i] + self.bi[j] + np.dot(p,q)
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

