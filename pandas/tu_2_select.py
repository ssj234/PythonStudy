#!/usr/bin/python
# -*- coding: UTF-8 -*-


# 
import numpy as np
import pandas as pd


dates = pd.date_range('20170301',periods=8)
df = pd.DataFrame(np.random.randn(8,5),index=dates,columns=list('ABCDE'))
print df


# select
# 打印A
print '\nselect column of A:\n'
print df['A']  # 选择A列
print '\ntype of column of A:\n'
print type(df['A']) # 选择A列的类型

print '\nselect row:\n'
print df[1:2] # 选择第1行
print '\ntype of row:\n'
print type(df[1:2])

print '\nselect year:\n'
print df['2017-03-01':'2017-03-03']  # 选择 [2017-03-01,2017-03-03] 包括左右范围内的数据

print '\nselect row and column:\n'
print df.loc['2017-03-01':'2017-03-03',['B','D']] # 选择[2017-03-01,2017-03-03]的B D 两列

 

