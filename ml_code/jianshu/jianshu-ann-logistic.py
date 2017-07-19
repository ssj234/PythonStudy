#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之2-Logistic分类函数
# http://www.jianshu.com/p/abc2acf092a3

import numpy as np
import matplotlib.pyplot as plt

# logistic函数
def logistic(z,derivate=False):
	if derivate == False:
		return 1 / (1 + np.exp(-z))
	else:
		return logistic(z) * (1-logistic(z))

# 等差数列 从-6到6之间，生成100个
z = np.linspace(-6,6,100)


# 绘制logistic和其导数
plt.plot(z,logistic(z),'b-')
plt.plot(z,logistic(z,True),'r-')
plt.xlabel('$z$',fontsize=15)
plt.ylabel('$\sigma(z)$',fontsize=15)
plt.title('logistic function')
plt.grid()
plt.show()


