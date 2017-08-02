#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# 泰坦尼克号
import numpy as np
import pandas as pd

# 第一步：读取数据
## 有行标和列表
## 从info()中可以看到age/embarked/home.dest/room/ticket/boat/6列数据有缺失
titanic = pd.read_csv('data/titanic.txt',index_col=0,header=0)
print titanic.info() # Pandas的DataFrame的info方法

## 根据经验选择特征
X = titanic.loc[:,['pclass','age','sex']]
y = titanic.loc[:,['survived']]
## 查看信息
print X.info()
print y.info()
## 填充age
X['age'].fillna(X['age'].mean(),inplace=True)
print X.info()

## 分割数据
from sklearn.cross_validation import train_test_split
# 第0列为列标，1-9为X数据 10为Y的值
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

## 由于pclass和sex不是float，使用DictVectorizer进行向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print vec.feature_names_

X_test = vec.fit_transform(X_test.to_dict(orient='record'))

# 开始训练
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)

# 使用分类报告器
from sklearn.metrics import classification_report
print '---------[DecisionTreeClassifier]-------------'
print classification_report(y_test,y_predict,target_names=['dided','survived'])

# 使用随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)

print '---------[RandomForestClassifier]-------------'
print classification_report(y_test,rfc_y_pred,target_names=['dided','survived'])

# 梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)
print '---------[GradientBoostingClassifier]-------------'
print classification_report(y_test,gbc_y_pred,target_names=['dided','survived'])

