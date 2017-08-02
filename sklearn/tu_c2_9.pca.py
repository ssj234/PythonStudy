#!/usr/bin/python
# -*- coding: UTF-8 -*-
matplot = True
# Python机器学习及实践的第二章中的例子，
# 使用手写数字，先对其将维，再训练
# 结论是虽然损失了2%的准确性，但是降低了68.7%的维度
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 导入数据，每条数据65个， 最后一个是结果
digits_train=pd.read_csv('data/optdigits.tra',header=None)
digits_test=pd.read_csv('data/optdigits.tes',header=None)
x_digits=digits_train[np.arange(64)]
y_digits=digits_train[64]


# 导入PCA
from sklearn.decomposition import PCA
estimator=PCA(n_components=2) # 将64维压缩到2维
x_pca=estimator.fit_transform(x_digits)


from matplotlib import pyplot as plt
def plot_pca_scatter():
	colors=['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']

	for i in range(len(colors)):
		px=x_pca[:,0][y_digits.as_matrix() == i]
		py=x_pca[:,1][y_digits.as_matrix() == i]
		plt.scatter(px,py,c=colors[i])
	plt.legend(np.arange(0,10).astype(str))
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()

if matplot == True:
	plot_pca_scatter()

X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
lsvc_y_predict=lsvc.predict(X_test)


from sklearn.metrics import classification_report
print '--------[LinearSVC][!pca]-----------'
print classification_report(y_test,lsvc_y_predict,target_names=np.arange(10).astype(str))


estimator=PCA(n_components=20) # 将64维压缩到20维
x_pca=estimator.fit_transform(x_digits)
pca_X_train = estimator.fit_transform(X_train)
pca_X_test = estimator.transform(X_test)

pca_svc = LinearSVC()
pca_svc.fit(pca_X_train,y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

print '--------[LinearSVC][pca=20]-----------'
print classification_report(y_test,pca_y_predict,target_names=np.arange(10).astype(str))

