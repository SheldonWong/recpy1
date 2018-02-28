易于理解的基于Python的推荐系统(评分预测)


主要包含：
1. data(数据集)
2. (datatype)
3. example(模型使用示例)
4. log(日志输出目录)
5. model(模型目录)
6. nlp(评论处理部分)
7. out(结果输出目录)
8. preprocess(数据读取与预处理)
9. test(测试目录)
10. utils(实用工具集)

模型部分目前包含：
MF(LMF-MF)
BMF(BiasedMatrixFactorization)
LBMF(Logistic BMF)
STMF(A Sentiment Topic Matrix Factorization for Recommendation)
LSTMF(Logistic STMF)
BLSTMF(Binary Logistic STMF)
DLSTMF(Deep Logistic STMF)




TODO:
0. 直接根据日志训练，而不是先转换成矩阵再训练，性能调优（已完成）
1. 数据结构设计（已完成）
2. TFIDF+AutoEncoder（已完成）
3. ...