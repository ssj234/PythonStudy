#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

count = 0
def arrayInfo(ndarray,info,before=None,binfo=None):
	global count
	print '\n------[%d]------' % count
	if before is not None:
		print binfo,before
	print '>>>',info
	print ndarray
	print 'type:',type(ndarray)
	if type(ndarray) == np.ndarray:
		print 'dtype:',ndarray.dtype
		print 'ndim:',ndarray.ndim
		print 'shape:',ndarray.shape
	print '---------------'
	count += 1

# 1.生成numpy的ndarray,
list = [[1,2,3,4]]
# 参数为：list或ndarray,以及dtype
nplst1 = np.array(list)
arrayInfo(nplst1,'np.array(list)')
nplst1 = np.array(list,dtype=np.str)
arrayInfo(nplst1,'np.array(list,dtype=np.str)')

# 类型转换,可以设置np.int32 或者 nplist1.shape
nplst1 = nplst1.astype(np.int32)
arrayInfo(nplst1,'nplst1.astype(np.int32)')

print '\n[][][][][][][][][][][][][][]'
count = 0

# 2.ndarray相加，会对每个元素进行相加
nplist1 = np.array([1,2,3,4])
nplist2 = np.array([9,8,7,6])
arrayInfo(nplist1 + nplist2,'ndarray add every element') 

print '\n[][][][][][][][][][][][][][]'
count = 0
# 3.生成数组
# np.empty()生成指定的矩阵，数值随机
nplist = np.empty([2,3])
arrayInfo(nplist,'np.empty create random number')

# np.zeros()生成指定的矩阵，数值为0
nplist = np.zeros([2,3])
arrayInfo(nplist,'np.zeros create number[0]')

# np.ones()生成指定的矩阵，数值为0
nplist = np.ones([2,3])
arrayInfo(nplist,'np.ones create number[1]')

# np.random.rand(2,4)生成指定的矩阵，数值为[0-1]
nplist = np.random.rand(2,3)
arrayInfo(nplist,'np.random.rand(2,3) create number[0-1]')

# np.random.randint(2,10,[2,3])生成指定的矩阵，数值为[0-1]
nplist = np.random.randint(2,10,[2,3])
arrayInfo(nplist,'np.random.randint(2,10,[2,3]) create number int[2-10]')

# np.random.randn(2,3)生成指定的矩阵，数值正态分布
nplist = np.random.randn(2,3)
arrayInfo(nplist,'np.random.randn(2,3) create number normal')

# np.random.choice([1,3,4,6,7])生成指定的矩阵，数值正态分布
nplist = np.random.choice([1,3,4,6,7],3)
arrayInfo(nplist,'np.random.choice([1,3,4,6,7]) choice from param[0]')

# np.arange(2,12) 生成等差数列，从2到11
nplist = np.arange(2,12)
arrayInfo(nplist,'np.arange(2,12) create number [2-11]')

print '\n[][][][][][][][][][][][][][]'
count = 0

# 4.numpy的ndarray函数
nplist = np.arange(0,12).reshape((2,2,3))
arrayInfo(nplist,'np.arange(0,12).reshape((2,2,3))')
print 'nplist.sum() = ' , nplist.sum()
print 'nplist.sum(axis=0) = \n' , nplist.sum(axis=0)
print 'nplist.sum(axis=1) = \n' , nplist.sum(axis=1)

# max-最大值 mean-平均数 min-最小值
# sqrt log exp sin

# 垂直连接 vstack有一个参数(n1,n2)
narr1 = np.array([1,2,3,4])
narr2 = np.array([5,6,7,8])
rs = np.vstack((narr1,narr2))
arrayInfo(rs,'np.vstack((narr1,narr2))')
# 水平连接 hstack有一个参数(n1,n2)
rs = np.hstack((narr1,narr2))
arrayInfo(rs,'np.hstack((narr1,narr2))')

# split拆分
narr1 = np.array([1,2,3,4])
rs = np.split(narr1,2)
arrayInfo(rs,'np.split(narr1,2)')

# where
nplist = np.array([1,1,1,0,0]) 
init = np.array([8,8,8,8,8])
zero = np.array([0,0,0,0,0])
nplist = np.where(nplist>0,init,zero) 
arrayInfo(nplist,'np.where(nplist>1,init,zero)',before=(nplist>1),binfo='cond: nplist>1 ')


print '\n[][][][][][][][][][][][][][]'
count = 0

# 5.numpy的线性代数
from numpy.linalg import inv,qr,det,eig
# 生成一个单位矩阵
arrayInfo(np.eye(3),'np.eye(3)')

# 生成一个单位矩阵
nplist = np.arange(1,10).reshape(3,3)
arrayInfo(nplist,'np.arange(1,7).reshape(2,3)')
print 'inverst:inv(nplist)',inv(nplist + np.eye(3))  	  # 求矩阵的逆
print 'np.transpose:\n',np.transpose(nplist)  # 求转置矩阵
print 'nplist.T\n',nplist.T       # 求转置矩阵
print 'det(nplist):',det(nplist)  # 行列式
print 'eig(nplist):',eig(nplist)  # 特征值和特征向量
