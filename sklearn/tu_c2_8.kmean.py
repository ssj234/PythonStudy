#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# 聚类：对手写数字进行聚类
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据：数据无行标和列标，最后一位为数字类型
digits_train=pd.read_csv('data/optdigits.tra',header=None)
digits_test=pd.read_csv('data/optdigits.tes',header=None)

## 分离64维度的像素特征与1维度的数字目标,每行数据65个 最后一个是数字
X_train=digits_train[np.arange(64)]
Y_train=digits_train[64]

X_test=digits_test[np.arange(64)]
Y_test=digits_test[64]


# 开始聚类
from sklearn.cluster import KMeans

## 初始化KMeans模型并设置聚类中心数量为10
kmeans=KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

# 计算度量
from sklearn import metrics
print metrics.adjusted_rand_score(Y_test,y_pred)


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,Y_train)
gbc_y_pred = gbc.predict(X_test)


from sklearn.metrics import classification_report
print '--------[GradientBoostingClassifier]-----------'
print classification_report(Y_test,gbc_y_pred,target_names=np.arange(10).astype(str))

