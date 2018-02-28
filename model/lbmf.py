#coding=utf8
import numpy as np
import math
from utils.logger import get_logger

logger = get_logger('e_lbmf_list')

'''
avg = (self.ave - 1.0)/4
global_bias = math.log(avg/(1-avg))
score = global_bias  + bu[i] + bi[j] + np.dot(self.u[i],self.v[j])
sig_score = 1.0 / (1+math.exp(-score))
rating_hat = ave/5 + sig_score * (max_rating - min_rating)
'''
class Lbmf:

    def __init__(self,a,testset,users,items,k,t,alpha,lamda):
        self.a = a
        self.testset = testset
        self.users = users
        self.items = items
        self.m = len(users)
        self.n = len(users)
        self.min_rating = 1.0
        self.max_rating = 5.0
        #这里如何区别开a的类型
        if(isinstance(a,np.ndarray)):
            self.ave = np.sum(a[a>0])/len(a[a>0])
        if(isinstance(a,list)):
            sum = 0.0
            for row in self.a:
                sum += row[1]
            self.ave = sum / len(a)
            avg = (self.ave - 1.0)/4
            self.global_bias = math.log(avg/(1-avg))
        self.t = t
        self.k = k
        self.alpha = alpha
        self.lamda = lamda

        self.bu = np.random.rand(self.m)
        self.bi = np.random.rand(self.n)
        self.u = np.random.rand(self.m,self.k)
        self.v = np.random.rand(self.n,self.k)
        
    def before_train(self):
        a = self.a
        ave = self.ave
        if(isinstance(a,np.ndarray)):
            offset = np.sum(np.fabs(a[a>0] - ave)) / len(a[a>0])
        if(isinstance(a,list)):
            sum = 0.0
            for row in a:
                sum += math.fabs(row[1] - ave)
            offset = sum / len(a)
        logger.info('parameter of tarin:'+'number of trainset='+str(len(a))+
            ',offset='+str(offset)+',global bias:'+str(self.global_bias)
            +',m='+str(self.m)+',n='+str(self.n)+',k='+str(self.k)) 
    '''
    @desc:直接通过日志训练，而不转换为矩阵在训练
    @param:
    @return:
    '''
    def train_by_log(self):
        self.before_train()
        for t in range(self.t):
            loss = 0.0
            for row in self.a:
                i = row[0][0]
                j = row[0][1]
                prediction = self.predict(i,j,True)
                err = row[1] - prediction
                for r in range(self.k):
                    gu = err * self.v[j][r] - self.lamda * self.u[i][r]
                    gv = err * self.u[i][r] - self.lamda * self.v[j][r]
                    self.u[i][r] += self.alpha * gu
                    self.v[j][r] += self.alpha * gv
                self.bu[i] += self.alpha * (err - self.lamda * self.bu[i])
                self.bi[j] += self.alpha * (err - self.lamda * self.bi[j])
                loss += err ** 2
            logger.info("t:%d====================loss:%f"%(t,loss))
            if(math.fabs(loss) < 150):
                break
        return self.u,self.v,self.bu,self.bi,self.global_bias       
    '''
    @desc:直接通过日志训练，而不转换为矩阵在训练
    @param:[index,(x,y),v]
    @return:
    '''
    def train_by_list(self):
        offset = self.before_train()
        for t in range(self.t):
            loss = 0.0
            for row in self.a:
                i = row[0][0]
                j = row[0][1]
                score = self.global_bias  + self.bu[i] + self.bi[j] + np.dot(self.u[i],self.v[j])
                sig_score = 1 / (1+ math.exp(-score))
                rating_hat = self.ave/5 + sig_score * (self.max_rating - self.min_rating)
                err = row[1] - rating_hat
                for r in range(self.k):
                    gu = err * sig_score * sig_score * (1 - sig_score) * self.v[j][r] - self.lamda * self.u[i][r]
                    gv = err * sig_score * sig_score * (1 - sig_score) * self.u[i][r] - self.lamda * self.v[j][r]
                    self.u[i][r] += self.alpha * gu
                    self.v[j][r] += self.alpha * gv
                self.bu[i] += self.alpha * (err * sig_score * sig_score * (1 - sig_score) - self.lamda * self.bu[i])
                self.bi[j] += self.alpha * (err * sig_score * sig_score * (1 - sig_score) - self.lamda * self.bi[j])
                loss += err ** 2
            prediction_list = self.prediction(self.testset)
            mse = self.evaluation(prediction_list)
            logger.info("t:%d====================alpha:%f,lamda:%f,loss:%f,mse:%f"%(t,self.alpha,self.lamda,loss,mse))
            if(math.fabs(loss) < 150):
                break
            #更新学习速率
            if(t % 5 == 0):
                self.alpha = 0.9 * self.alpha
        return self.u,self.v,self.bu,self.bi,self.global_bias       


    '''
    @desc:训练
    @param:self
    @return:u,v(m*k和n*k的矩阵)
    '''
    def train(self):
        self.before_train()
        for t in range(self.t):
            loss = 0.0
            for i in range(self.m):
                for j in range(self.n):
                    if(math.fabs(self.a[i][j]) > 1e-4):
                        err = self.a[i][j] - self.ave - self.bu[i] - self.bi[j] - np.dot(self.u[i],self.v[j])
                        for r in range(self.k):
                            gu = err * self.v[j][r] - self.lamda * self.u[i][r]
                            gv = err * self.u[i][r] - self.lamda * self.v[j][r]
                            self.u[i][r] += self.alpha * gu
                            self.v[j][r] += self.alpha * gv
                        self.bu[i] += self.alpha * (err - self.lamda * self.bu[i])
                        self.bi[j] += self.alpha * (err - self.lamda * self.bi[j])
                        loss += err ** 2
            logger.info("t:%d====================loss:%f"%(t,loss))
            if(math.fabs(loss) < 10):
                break
        return self.u,self.v,self.bu,self.bi,self.ave

    '''
    @desc: 获取预测评分列表
    @param: u,v 
    @return: 预测评分列表
    '''
    def predict(self,i,j,bound):

        score = self.global_bias  + self.bu[i] + self.bi[j] + np.dot(self.u[i],self.v[j])
        sig_score = 1.0 / (1+math.exp(-score))
        rating_hat = self.ave/5 + sig_score * (self.max_rating - self.min_rating)

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
        mse = np.sum( (prediction[:,3] - prediction[:,2])**2 ) / len(prediction)
        return mse 

    '''
    @desc:求导
    @param:目标函数
    @return:偏导
    ''' 