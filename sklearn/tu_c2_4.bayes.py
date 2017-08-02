#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# 新闻分类

# 第一步：从sklearn获取内置的数据
from sklearn.datasets import fetch_20newsgroups
## 获取数据
news=fetch_20newsgroups(subset='all')
## 查看数据细节
print len(news.data) # 18846
print news.data[0] # 

## 拆分数据
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

## 文件特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 第二步，开始训练
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)

# 测量性能
from sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=news.target_names)
