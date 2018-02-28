Google python风格指南
命名规范中，除了类、Exception使用驼峰，GLOBAL_VAR_NAME使用“全大写加下划线”
其他基本都是“小写和下划线”的组合

这里包含使用各种模型的示例：
1. e_MF.py
2. e_BMF.py
3. e_Arts(使用Arts数据集的示例)

基本步骤
1. 读取数据
2. 预处理（去除重复元素，编码user_id 与 item_id）
3. 切分数据
4. 计算主题
5. 计算情感
6. 构建字典user-pref
7. 构建字典item-vertor
8. 训练 k t alpha lamda
9. 预测及评测


