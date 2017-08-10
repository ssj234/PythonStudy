#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习视频中的例子，特征选择
# 观察特征对结果的影响程度
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 第一步：读取数据
## 有行标和列表
## 从info()中可以看到age/embarked/home.dest/room/ticket/boat/6列数据有缺失
titanic = pd.read_csv('data/titanic.txt',index_col=0,header=0)
print titanic.info() # Pandas的DataFrame的info方法

titanic['age'].fillna(titanic['age'].mean(),inplace=True)
titanic['embarked'].fillna('Southampton',inplace=True)
titanic['home.dest'].fillna('0',inplace=True)

titanic.loc[titanic['sex'] == 'male', 'sex'] = 0
titanic.loc[titanic['sex'] == 'female', 'sex'] = 1
titanic.loc[titanic['embarked'] == 'Southampton', 'embarked'] = 0
titanic.loc[titanic['embarked'] == 'Cherbourg', 'embarked'] = 1
titanic.loc[titanic['embarked'] == 'Queenstown', 'embarked'] = 2

titanic.loc[titanic['pclass'] == '1st', 'pclass'] = 0
titanic.loc[titanic['pclass'] == '2nd', 'pclass'] = 1
titanic.loc[titanic['pclass'] == '3rd', 'pclass'] = 2

titanic['namelength'] = titanic['name'].apply(lambda x: len(x))
titanic['homelength'] = titanic['home.dest'].apply(lambda x: len(x))


print titanic.info()

predictions = ['pclass','sex','age','embarked','namelength','homelength']


# 导入SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif,k=2)
selector.fit(titanic[predictions],titanic['survived'])
scores = -np.log10(selector.pvalues_)
# 展示图像
plt.bar(range(len(predictions)),scores)
plt.xticks(range(len(predictions)),predictions,rotation='vertical')
plt.show()