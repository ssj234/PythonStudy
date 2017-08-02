#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# K近邻：设置的K不同，预测的分类也不有所区别

# 第一步：从sklearn获取内置的数据
from sklearn.datasets import load_iris
## 获取数据
iris = load_iris()
## 查看数据细节
print iris.data.shape
print iris.DESCR

## 拆分数据
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

## 标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 第二步，开始训练，knn需要对k的值进行变化并测量其性能
for size in range(1,40):
	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors=size)
	knn.fit(X_train,y_train)
	y_predict = knn.predict(X_test)

	# 第三步 测量性能
	from sklearn.metrics import classification_report
	print '[k=%d]accuracy: %f' % (size,knn.score(X_test,y_test))
# print classification_report(y_test,y_predict,target_names=iris.target_names)
