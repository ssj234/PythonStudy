#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
plt.xlabel('Thickness')
plt.ylabel('Cell Size')
# plt.show()

# 绘制测试数据
plt.subplot(2,2,2)
plt.scatter(df_test[df_test['Type']==0].iloc[:,0],df_test[df_test['Type']==0].iloc[:,1],marker='o',s=20,c='red')
plt.scatter(df_test[df_test['Type']==1].iloc[:,0],df_test[df_test['Type']==1].iloc[:,0],marker='x',s=10,c='blue')
plt.xlabel('Thickness')
plt.ylabel('Cell Size')
# plt.show()



# 线性回归进行分类
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(trainX,trainY)

rd0 = np.linspace(1,10,10).reshape(-1,1)
rd1 = np.linspace(1,10,10).reshape(-1,1)
rdX = np.hstack((rd0,rd1))
rdY = lr.predict(rdX).reshape(-1,1)
print rdX
print rdY

df = pd.DataFrame(np.hstack((rdX,rdY)),columns=['X','Y','T'])

plt.subplot(2,2,3)
plt.scatter(df[df['T']==0].iloc[:,0],df[df['T']==0].iloc[:,1],marker='o',s=20,c='red')
plt.scatter(df[df['T']==1].iloc[:,0],df[df['T']==1].iloc[:,1],marker='o',s=10,c='blue')
plt.xlabel('Thickness')
plt.ylabel('Cell Size')
plt.show()


