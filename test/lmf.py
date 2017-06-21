import sys
sys.path.append('../')
from preprocess import preprocess
import numpy as np 


ratings_df = preprocess.readdata('j:/amazon/output/Furniture@uir.csv',',')

ratings_r_df,users,items = preprocess.replace_user_and_item(ratings_df)

l = []
#row[0]=index,row[1]=user_id,row[2]=item_id
for row in ratings_r_df.itertuples():
    t = [(row[1],row[2]),row[3]]
    l.append(t)	

row = l[0]
print(row)
print(row[0][0],row[0][1],row[1])

'''
@desc:训练
@param:trainMatrix,k(int,代表factors数量)
@return:u,v(m*k和n*k的矩阵)
'''
def train(l,k):
    m = len(users)
    n = len(items)
    alpha = 0.01
    lamda = 0.01
    u = np.random.rand(m,k)
    v = np.random.rand(n,k)
    print('parameter of tarin:',m,n,np.shape(u),np.shape(v))
    
    for t in range(60):
        loss = 0.0
        for row in l:
            i = row[0][0]
            j = row[0][1]
            err = row[1] - np.dot(u[i],v[j])
            for r in range(k):
                gu = err * v[j][r] - lamda * u [i][r]
                gv = err * u[i][r] - lamda * v [j][r]
                u[i][r] += alpha * gu
                v[j][r] += alpha * gv
            loss += err ** 2
        print("t:%d====================loss:%f"%(t,loss))
    return u,v

u,v = train(l,5)



'''
#验证代码
x = ratings_r['user_id'][0]
y = ratings_r['item_id'][0]
print(users[x])
print(items[y])
'''


