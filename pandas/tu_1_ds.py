#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

# Series
s = pd.Series([i*2 for i in range(1,11)])
print '\npd.Series([i*2 for i in range(1,11)])\n'
print s
print '\ntype(s)\n'
print type(s)

s = pd.Series([1,2,3,4],index=list('ABCD'))
print "\npd.Series([1,2,3,4],index=list('ABCD'))\n"
print s

# Series中选择
print s[s>1]
print s *2


# 创建DataFrame 单独设置每列的数据 
# A都是1  B都是日期   C都是1并设置了列标  D为list[3,3,3,3]
df = pd.DataFrame({"A":1,"B":pd.Timestamp('20170301'),"C":pd.Series(1,index=list(range(10,14)),dtype='float32'),"D":np.array([3]*4,dtype="float32")})
print '\npd.DataFrame:\n'
print df

# 创建DataFrame，参数为：ndarray 一个矩阵，index=行标  column=列标
dates = pd.date_range('20170301',periods=8)
df = pd.DataFrame(np.random.randn(8,5),index=dates,columns=list('ABCDE'))
print df

# Basic Operation
print '\nhead(3)\n'
print df.head(3)  # 打印前三行
print '\ntail(3)\n'
print df.tail(3)  # 打印后三行
print '\ndf.index\n'
print df.index    # 打印行标
print '\ndf.values\n'
print df.values   # 打印内容
print '\ndf.T\n' 
print df.T   #转置
print '\ndf.describe\n'
print df.describe()  # 打印数据描述 最大值 最小 平均值  分布等

# select

nadf = pd.DataFrame(np.arange(12).reshape(3,4),index=['one','two','three'],columns=['A','B','C','D'])
print nadf

# 使用切片
print nadf[0:2]   # 使用下标 必须使用:
print nadf['one':'two']   # 使用行标,必须使用:

# 使用loc 按照行标切片  行标包括结尾
print nadf.loc['one':'two'] # 两个参数，第一个是行,必须使用行标，第二个是列
print nadf.loc['one':'two','B':'D'] # 两个参数，第一个是行,必须使用行标，第二个是列
print nadf.loc[['one','three'],['B','D']] # 两个参数，可以是列表，也可以是:的切片

# 使用iloc 按照下标切片  下标不包括结尾
print nadf.iloc[0:2] #
print nadf.iloc[0:2,2:4]  # 获取0和1行，2和3列
print nadf.iloc[[1,2],[0,2]]  # 获取1和2行 0和2列

# 按照条件选择
print nadf[nadf>3] # 大于3的去原来的值，否则为NaN
nadfb = nadf[nadf>3]

# 添加列
nadf['F'] = pd.Series([91,92,93],index=['one','two','three'])

# 修改列
nadf['F'] = 1

# 缺失值处理
# dropna或者fillna
print nadfb
print nadfb.dropna()  # 丢弃NaN的值
print nadfb.fillna(1) # 填充值












