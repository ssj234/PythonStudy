#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

# 加载数据
iris = load_iris()
# iris.data 类型为ndarray
# print iris.data     # 数据
# print iris.target   # 结果

trainX,testX,trainY,testY = train_test_split(iris.data,iris.target,test_size=0.2,random_state=1)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(trainX,trainY)
y_pred = clf.predict(testX)

# verify
from sklearn import metrics

# 判断成功率
print metrics.accuracy_score(y_true=testY,y_pred=y_pred)

# 混淆矩阵
print metrics.confusion_matrix(y_true=testY,y_pred=y_pred)

# 理想情况下是一个对角阵
# 横轴 实际值 
# [[11  0  0]  # 实际是1类  
# [ 0 12  1]   # 实际是2类 有一个预测错了
# [ 0  0  6]]

# 将决策树的值保存到文件

with open('./out/tree.dot','w') as fw:
	tree.export_graphviz(clf,out_file=fw)