#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，根据肿瘤的多个维度对良性/恶性进行判断

matplot = True # 是否绘制图形

import pandas as pd
import numpy as np

# 第一步，分析数据，文件中的数据 
# 有行标，无列标，
# X有9个维度 Y为2-良性 4-恶性 
# 缺失值使用?表示
# 二分类问题

## 定义列标
column_names=['sample code number','clump tickness','uniformity of cell size',
'uniformity of cell shape','marginal adhesion','single epithelial cell size',
'bare nuclei','bland chromatin','normal nucleoli','mitoses','class']

## 读取数据集并处理
data=pd.read_csv('data/breast-cancer-wisconsin.data',names=column_names)
## 数据处理:替换缺失值，将?替换为nan
data=data.replace(to_replace='?',value=np.nan)
## 数据处理:删除缺失值，删除为nan的值
data=data.dropna(how='any')
## 输出数据的数据量和维度
print data.shape

## 拆分数据集
#使用sklearn的train_test_split模块用于分割数据
from sklearn.cross_validation import train_test_split
#随机采样 25%用于测试 75%用于训练集合
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

print y_train.value_counts() # 查看训练样本的数量和类别分布
print y_test.value_counts() # 查看测试样本的数量和类别分布


## 标准化数据
from sklearn.preprocessing import StandardScaler
## 准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不被某些维度过大的特征值而主导
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


# 第二步：开始训练

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
lr=LogisticRegression() # Logistic分类
sgdc=SGDClassifier() # 使用随机梯度下降的线性分类

## 调用LogisticRegression中的fit函数/模块来训练模型参数，并预测测试集
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
## 调用随机梯度估计来训练模型，并预测测试集
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)


## 第三步：性能分析
from sklearn.metrics import classification_report

## 使用score方法和classification_report评判
print 'Accuracy of LR Classifier:',lr.score(X_test,y_test)
print classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])
print 'Accuracy of SGD Classifier:',sgdc.score(X_test,y_test)
print classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])

from sklearn import metrics
## 判断成功率
print metrics.accuracy_score(y_true=y_test,y_pred=lr_y_predict)
print metrics.accuracy_score(y_true=y_test,y_pred=sgdc_y_predict)
## 混淆矩阵
print metrics.confusion_matrix(y_true=y_test,y_pred=lr_y_predict)
print metrics.confusion_matrix(y_true=y_test,y_pred=sgdc_y_predict)