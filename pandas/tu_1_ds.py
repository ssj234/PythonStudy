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

print df.iloc[1:3,2:4] # 选择12行 23列
print df.loc['2017-03-01':'2017-03-04',['B','D']] # 4行2列
# df[row][column]  先选择行 再选择列
# DataFrame 的每个列是一个Series

# 重新索引

obj = pd.Series([4.5,7.2,-5.3,3.6],index = ['d','b','a','c'])
obj2 = obj.reindex(['a','b','c','d']) # 修改Series的索引顺序
# 除了修改索引顺序 还可以添加或删除行标，默认赋值fill_value=0.0
obj2 = obj.reindex(['a','b','c','d','e','f'],fill_value=0.0)


















