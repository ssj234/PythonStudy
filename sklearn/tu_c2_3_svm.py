#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# 手写字体识别

matplot = True # 是否绘制图形

# 第一步：分析数据
## 导入数据：从sklearn获取内置的数据
from sklearn.datasets import load_digits
digits=load_digits() # 获取数据
print digits.data.shape # 获取数据的描述（行和列）


## 分割数据
from sklearn.cross_validation import train_test_split
### 随机采样 25%用于测试 75%用于训练集合
### digits.target是结果集 X是特征向量 Y是结果集合
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

print y_train.shape #查看训练样本的数量和类别分布
print y_test.shape #查看测试样本的数量和类别分布

## 标准化数据
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

# 第二步：训练数据
from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
y_predict=lsvc.predict(X_test)


# 第三步：性能测评
print 'The Accuracy of Linear SVC is',lsvc.score(X_test,y_test)
from sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))