#coding=utf8
import numpy as np
import math
from utils.logger import get_logger

logger = get_logger('bmf')
'''
rating_hat = mu + bu[i] + bi[j] + np.dot(u,v)

'''
class Bmf:

    def __init__(self,a,testset,users,items,k,t,alpha,lamda):
        self.a = a
        self.testset = testset
        self.users = users
        self.items = items
        self.m = len(users)
        self.n = len(users)
        
        if(isinstance(a,np.ndarray)):
            self.mu = np.sum(a[a>0])/len(a[a>0])
        if(isinstance(a,list)):
            sum = 0.0
            for row in self.a:
                sum += row[1]
            self.mu = sum / len(a)
        
        
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
        if(isinstance(a,np.ndarray)):
            offset = np.sum(np.fabs(a[a>0] - self.mu)) / len(a[a>0])
        if(isinstance(a,list)):
            sum = 0.0
            for row in a:
                sum += math.fabs(row[1] - self.mu)
            offset = sum / len(a)
        logger.info('parameter of tarin:'+'number of trainset='+str(len(a))+
            ',offset='+str(offset)+',m='+str(self.m)+',n='+str(self.n)+',k='+str(self.k))   
    '''
    @desc:直接通过日志训练，而不转换为矩阵在训练
    @param:
    @return:
    '''
    def train_by_list(self):
        self.before_train()
        for t in range(self.t):
            loss = 0.0
            for row in self.a:
                i = row[0][0]
                j = row[0][1]
                prediction = self.predict(i,j,False)
                err = row[1] - prediction
                for r in range(self.k):
                    gu = err * self.v[j][r] - self.lamda * self.u[i][r]
                    gv = err * self.u[i][r] - self.lamda * self.v[j][r]
                    self.u[i][r] += self.alpha * gu
                    self.v[j][r] += self.alpha * gv
                self.bu[i] += self.alpha * (err - self.lamda * self.bu[i])
                self.bi[j] += self.alpha * (err - self.lamda * self.bi[j])
                loss += err ** 2
            prediction_list = self.prediction(self.testset)
            mse = self.evaluation(prediction_list)
            logger.info("t:%d====================loss:%f,mse:%f"%(t,loss,mse))
            if(math.fabs(loss) < 150):
                break
        return self.u,self.v,self.bu,self.bi,self.mu   
    


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
        return self.u,self.v,self.bu,self.bi,self.mu


    def predict(self,i,j,bound=True):
        rating_hat = self.mu  + self.bu[i] + self.bi[j] + np.dot(self.u[i],self.v[j])
        if(bound):
            if (rating_hat > 5):
                return 5
            if (rating_hat < 1):
                return 5
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