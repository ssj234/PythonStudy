#!/usr/bin/python
# -*- coding: UTF-8 -*-

#假设要学习到的最后的方程为y=x^2-0.5 

#构造数据
import numpy as np

# 构造300个点，分布在-1到1区间 
# linspace函数通过指定开始值、终值和元素个数来创建一维数组，等差数列
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 +noise

# 输出数据
print x_data.shape,y_data.shape

import matplotlib.pyplot as plt

# 分割出3行2列子图，在1号作图
plt.subplot(1,1,1)
plt.title('y=x^2-0.5')
plt1,=plt.plot(x_data,y_data,color='red',label='train data(noise)')   



# 使用线性进行拟合
from sklearn.linear_model import LinearRegression


# 处理为2次方
# 专门产生多项式的，并且多项式包含的是相互影响的特征集。
# 比如：一个输入样本是２维的。形式如[a,b] ,则二阶多项式的特征集如下[1,a,b,a^2,ab,b^2]。
from sklearn.preprocessing import PolynomialFeatures
poly2=PolynomialFeatures(degree=2)
# [3] --> [ 1.,  3.,  9.]
x_data_poly2=poly2.fit_transform(x_data)


lr=LinearRegression()
lr_ploy=LinearRegression()
lr.fit(x_data,y_data)
lr_ploy.fit(x_data_poly2,y_data)


x_test=np.linspace(-1,1,100)[:,np.newaxis]
lrResult=lr.predict(x_test)
lrPloy=lr_ploy.predict(poly2.fit_transform(x_test))
rs_real=np.square(x_test)-0.5

# 绘制3条曲线

plt2,=plt.plot(x_test,rs_real,color='blue',label='real')   
plt3,=plt.plot(x_test,lrResult,color='black',label='linear')   
plt4,=plt.plot(x_test,lrPloy,color='yellow',label='ploy')   
plt.legend(handles=[plt1,plt2,plt3,plt4])
plt.show()