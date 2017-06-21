#coding=utf8

def desc():
    #评分分布
    one = len([o for o in ratings['rating'] if o == 1])
    two = len([t for t in ratings['rating'] if t == 2])
    three = len([th for th in ratings['rating'] if th == 3])
    four = len([f for f in ratings['rating'] if f == 4])
    five = len([fi for fi in ratings['rating'] if fi == 5])
    #均值
    ave = sum(ratings['rating'])/len(ratings['rating'])
    small = [a for a in ratings['rating'] if a < ave]
    big = [b for b in ratings['rating'] if b > ave]
    #总体方差
    variance = sum((ratings['rating'] - ave)**2)/len(ratings['rating'])
    #平均绝对偏差，mean absolute error
    mae = sum(np.abs(ratings['rating'] - ave))/len(ratings['rating'])

    rating_number = len(ratings)
    user_number = len(list(set(ratings['user_id'])))
    item_number = len(list(set(ratings['item_id'])))
    print('数据总量:',rating_number)
    print('用户数量:',user_number)
    print('物品数量:',item_number)
    print('1分数量:',one)
    print('2分数量:',two)
    print('3分数量:',three)
    print('4分数量:',four)
    print('5分数量:',five)
    print('稀疏率:',1-(rating_number/(user_number * item_number)))
    print('平均值:',ave)
    print('评分小于平均值的数量:',len(small))
    print('评分大于平均值的数量:',len(big))
    print('总体方差:',variance)
    print('平均绝对偏',mae)

'''
================Arts.txt上的执行结果====================
数据总量: 27980
用户数量: 24071
物品数量: 4211
1分数量: 2831
2分数量: 1434
3分数量: 2219
4分数量: 4687
5分数量: 16809
稀疏率: 0.9997239623408471
平均值: 4.1154038599
评分小于平均值的数量: 11171
评分大于平均值的数量: 16809
总体方差: 1.78235743161
平均绝对偏 1.06284321079
[Finished in 1.1s]
'''