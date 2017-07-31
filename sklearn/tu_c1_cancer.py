#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm

# 数据来源,训练数据和测试数据分布从文件中读取
## 首先要分析数据，有行标和列表 ，X有2纬，最后一列为y
df_train = pd.read_csv('./data/breast-cancer-train.csv',index_col=0)
df_test = pd.read_csv('./data/breast-cancer-test.csv',index_col=0)

## 拆分X和Y
trainX = df_train.iloc[:,[0,1]]
trainY = df_train.iloc[:,[2]]

testX = df_test.iloc[:,[0,1]]
testY = df_test.iloc[:,[2]]


# print df_train[df_train['Type']==0].iloc[:,0]
# print df_train[df_train['Type']==0].iloc[:,1]

# 绘制训练的数据
plt.subplot(2,2,1)
plt.scatter(df_train[df_train['Type']==0].iloc[:,0],df_train[df_train['Type']==0].iloc[:,1],marker='o',s=20,c='red')
plt.scatter(df_train[df_train['Type']==1].iloc[:,0],df_train[df_train['Type']==1].iloc[:,0],marker='x',s=10,c='blue')
plt.xlabel('Train-Thickness')
plt.ylabel('Train-Cell Size')
# plt.show()

# 绘制测试数据
plt.subplot(2,2,2)
plt.scatter(df_test[df_test['Type']==0].iloc[:,0],df_test[df_test['Type']==0].iloc[:,1],marker='o',s=20,c='red')
plt.scatter(df_test[df_test['Type']==1].iloc[:,0],df_test[df_test['Type']==1].iloc[:,0],marker='x',s=10,c='blue')
plt.xlabel('test-Thickness')
plt.ylabel('test-Cell Size')
# plt.show()



# 线性回归进行分类
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(trainX,trainY)

# 预测测试集并计算正确率
predY = lr.predict(testX)
## 导入度量类
from sklearn import metrics
## 判断成功率
print metrics.accuracy_score(y_true=testY,y_pred=predY)
## 混淆矩阵
print metrics.confusion_matrix(y_true=testY,y_pred=predY)


# 绘制区域，创建100个点，从1到10等差数列
nb_of_xs = 100
rd0 = np.linspace(1,10,num=nb_of_xs).reshape(-1,1)
rd1 = np.linspace(1,10,num=nb_of_xs).reshape(-1,1)
xx, yy = np.meshgrid(rd0, rd1) # create the grid
## 创建颜色映射
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])
## 进行预测，保存预测值
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i,j] = lr.predict(np.asmatrix([xx[i,j], yy[i,j]]))

plt.subplot(2,2,3)
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.xlabel('Thickness')
plt.ylabel('Cell Size')
plt.show()




