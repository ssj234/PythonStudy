#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# 波士顿房价
import numpy as np
# 第一步：导入数据
from sklearn.datasets import load_boston
boston=load_boston()
print boston.DESCR # 查看数据格式

## 拆分数据
from sklearn.cross_validation import train_test_split
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

print 'The max value is ',np.max(boston.target)
print 'The min value is ',np.min(boston.target)
print 'The avg value is ',np.mean(boston.target)

## 标准化数据
from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_Y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_Y.fit_transform(y_train)
y_test=ss_Y.transform(y_test)

# 开始训练
## 使用线性回归
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)


## 使用随机梯度下降 
from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict=sgdr.predict(X_test)




# 查看性能
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

print '-----------[SGDRegressor]-----------'
print 'LinearRegression default',lr.score(X_test,y_test)
print 'LinearRegression R-squared',r2_score(y_test,lr_y_predict)
print 'LinearRegression MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(lr_y_predict))
print 'LinearRegression MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(lr_y_predict))

# SGD
print '-----------[SGDRegressor]-----------'
print 'SGDRegressor default',sgdr.score(X_test,y_test)
print 'SGDRegressor R-squared',r2_score(y_test,sgdr_y_predict)
print 'SGDRegressor MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(sgdr_y_predict))
print 'SGDRegressor MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(sgdr_y_predict))

# 使用SVM回归
from sklearn.svm import SVR
## 使用线性核函数配置的支持向量机进行回归训练
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

## 使用多项式核函数配置的支持向量机进行回归训练
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

## 使用径向核函数配置的支持向量机进行回归训练
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)


# linear
print '-----------[kernel="linear"]-----------'
print 'kernel="linear" default',linear_svr.score(X_test,y_test)
print 'kernel="linear" R-squared',r2_score(y_test,linear_svr_y_predict)
print 'kernel="linear" MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(linear_svr_y_predict))
print 'kernel="linear" MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(linear_svr_y_predict))


# poly
print '-----------[kernel="poly"]-----------'
print 'kernel="poly" default',poly_svr.score(X_test,y_test)
print 'kernel="poly" R-squared',r2_score(y_test,poly_svr_y_predict)
print 'kernel="poly" MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(poly_svr_y_predict))
print 'kernel="poly" MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(poly_svr_y_predict))


# rbf
print '-----------[kernel="rbf"]-----------'
print 'kernel="rbf" default',rbf_svr.score(X_test,y_test)
print 'kernel="rbf" R-squared',r2_score(y_test,rbf_svr_y_predict)
print 'kernel="rbf" MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(rbf_svr_y_predict))
print 'kernel="rbf" MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(rbf_svr_y_predict))


# K近邻
from sklearn.neighbors import KNeighborsRegressor
## 初始化K近邻回归器，预测方式为平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

## 初始化K近邻回归器，预测方式为距离加权回归
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# uniform
print '-----------[KNeighborsRegressor="uniform"]-----------'
print 'KNeighborsRegressor="uniform" default',uni_knr.score(X_test,y_test)
print 'KNeighborsRegressor="uniform" R-squared',r2_score(y_test,uni_knr_y_predict)
print 'KNeighborsRegressor="uniform" MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(uni_knr_y_predict))
print 'KNeighborsRegressor="uniform" MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(uni_knr_y_predict))


# distance
print '-----------[KNeighborsRegressor="distance"]-----------'
print 'KNeighborsRegressor="distance" default',dis_knr.score(X_test,y_test)
print 'KNeighborsRegressor="distance" R-squared',r2_score(y_test,dis_knr_y_predict)
print 'KNeighborsRegressor="distance" MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(dis_knr_y_predict))
print 'KNeighborsRegressor="distance" MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(dis_knr_y_predict))



# 回归树
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict = dtr.predict(X_test)

# distance
print '-----------[DecisionTreeRegressor]-----------'
print 'DecisionTreeRegressor',dtr.score(X_test,y_test)
print 'DecisionTreeRegressor',r2_score(y_test,dtr_y_predict)
print 'DecisionTreeRegressor',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(dtr_y_predict))
print 'DecisionTreeRegressor',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(dtr_y_predict))


# 集成模型
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict = rfr.predict(X_test)



etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict = etr.predict(X_test)


gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict = gbr.predict(X_test)


# RandomForestRegressor
print '-----------[RandomForestRegressor]-----------'
print 'RandomForestRegressor',rfr.score(X_test,y_test)
print 'RandomForestRegressor',r2_score(y_test,rfr_y_predict)
print 'RandomForestRegressor',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(rfr_y_predict))
print 'RandomForestRegressor',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(rfr_y_predict))


# ExtraTreesRegressor
print '-----------[ExtraTreesRegressor]-----------'
print 'ExtraTreesRegressor',etr.score(X_test,y_test)
print 'ExtraTreesRegressor',r2_score(y_test,etr_y_predict)
print 'ExtraTreesRegressor',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(etr_y_predict))
print 'ExtraTreesRegressor',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(etr_y_predict))


# GradientBoostingRegressor
print '-----------[GradientBoostingRegressor]-----------'
print 'GradientBoostingRegressor',gbr.score(X_test,y_test)
print 'GradientBoostingRegressor',r2_score(y_test,gbr_y_predict)
print 'GradientBoostingRegressor',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(gbr_y_predict))
print 'GradientBoostingRegressor',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(gbr_y_predict))
