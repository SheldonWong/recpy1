import pandas as pd 

'''
主要用来构建user-pref和item-topic
'''


#构建两个字典，分别是用户-情感向量，物品-topic向量

#1. 构建字典user-pref矩阵
uisv_names = ['user','item','sentiment','vector','pref']
uisv_df = pd.read_csv('../data/out/uisv.csv',header=None,names=uisv_names)
user_l = uisv_df['user'].tolist()
pref_l = uisv_df['pref'].tolist()


u_dict = dict(zip(user_l,pref_l))


#2 构建item-vertor矩阵
doc_topic_df = pd.read_csv('../data/out/doc_topic.csv',header=None)
item_review_df = pd.read_csv('../data/out/item_review.csv',header=None)

item_id_l = list(item_review_df[0])
doc_topic_l = doc_topic_df.as_matrix().tolist()

v_dict = dict(zip(item_id_l,doc_topic_l))
print(len(v_dict))
print(v_dict.get(701))




'''
#构建物品topic向量
item_df = pd.read_csv('../data/out/item_review.csv',header=None)
vector_df =  pd.read_csv('../data/out/doc_topic.csv',header=None)

#print(list(item_df[0]))
#print(vector_df.as_matrix().tolist())

item_l = list(item_df[0])
vector_l = vector_df.as_matrix().tolist()

item_vector = zip(item_l,vector_l)

item_vector_dict = dict((name,value) for name,value in item_vector)

print(dir(item_vector_dict))
print(len(item_vector_dict))
print(item_vector_dict.get(701))
'''