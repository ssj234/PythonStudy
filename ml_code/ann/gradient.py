#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# 梯度下降算法
begin = 0
index = 0
current = 0
rate = 0.01
# 构造100个点，分布在-1到1区间 
# linspace函数通过指定开始值、终值和元素个数来创建一维数组，等差数列
x_data=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data = 2*(x_data)+ 0.05+noise


# 生成随机权重
weights = np.random.rand(2)
print weights

def h(x_ndarray):
    return x_ndarray*weights[0]+weights[1]

while index < 100:
    current = 0
    while current < len(x_data):
        x = x_data[current]
        y = y_data[current]
        loss = y-h(x)
        sum_a =  rate*(loss)*x[0]
        sum_b =  rate*(loss)
        #print '>>>',weights,weights[0]*x+weights[1]
        #print x,y,h(x),y-h(x)
        weights[0] = weights[0] + sum_a
        weights[1] = weights[1] + sum_b
        current += 1
    print index,current,weights,loss
    index += 1
        
print weights



# 分割出1行1列子图，在1号作图
plt.subplot(1,1,1)
plt.title('y=x*2')
plt1,=plt.plot(x_data,y_data,'o',color='red',label='train data(noise)')   

x_test=np.linspace(-1,1,100)[:,np.newaxis]
rs_real=x_test*2+ 0.05
y_test = x_test * weights[0] +weights[1]


plt2,=plt.plot(x_test,rs_real,color='blue',label='real')   
plt3,=plt.plot(x_test,y_test.reshape(-1,1),color='yellow',label='Gradient')   
# plt4,=plt.plot(x_test,lrPloy,color='yellow',label='ploy')   
plt.legend(handles=[plt1,plt2,plt3])
plt.show()