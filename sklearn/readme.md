# Python机器学习及实践

## 主要流程

1. 分析训练数据和测试数据，主要是对数据的分布和特征意义进行总结和处理。如选取有意义的特征，填充缺失值，正则化，归一化，二次项等操作，还可以使用绘图工具查看数据分布情况。

* 查看数据分布：

```
y_train.value_counts() # 查看每个数值的数量
```

* 拆分数据集合：

```
from sklearn.cross_validation import train_test_split
# 第0列为列标，1-9为X数据 10为Y的值
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
```

* 标准化数据:保证每个维度的特征数据方差为1，均值为0，使得预测结果不被某些维度过大的特征值而主导,公式为：(X-mean)/std，减去平均值后除以方差

```
from sklearn.preprocessing import StandardScaler
#标准化数据，
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
```

2. 使用分类或回归工具进行训练,常用的工具有：
分类包括二分类和多分类两种，如良性/恶性肿瘤是二分类问题，手写数字识别是多分类问题。

```
# logistic回归分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)


# 随机梯度下降 Stochastic Gradient Descent.
# 为了更好的训练效果，SDG分类器需要对数据进行标准化（平均值为0，方差为1）
from sklearn.linear_model import SGDClassifier
sgdc=SGDClassifier() # 使用随机梯度下降的线性分类
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)

# SVM
from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
y_predict=lsvc.predict(X_test)

# 朴素贝叶斯

```

3. 在测试集上，对模型进行度量，分类和回顾的度量方法不同

* 对于分类问题
* 准确率Accuracy = 预测正确的数量/所有数量
* 精确率Precision = 预测正确的数量/预测正确的数量 + 假阳性（误诊）
* 召回率Recall = 预测正确的数量/预测正确的数量 + 假阴性（漏诊）
* F1 = 2 / ((1/Precision) + (1/Recall))

```
from sklearn import metrics
## 判断成功率
print metrics.accuracy_score(y_true=testY,y_pred=predY)
## 混淆矩阵
print metrics.confusion_matrix(y_true=testY,y_pred=predY)

# 使用分类报告器
from sklearn.metrics import classification_report
classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])

# 使用分类器的score方法
lr=LogisticRegression() 
lr.score(X_test,y_test)
```

* 对于回归问题

## 监督学习
### 分类学习
### 回归预测

## 无监督学习
### 数据聚类
### 特征降纬

## 实用技巧

### 特征提升
### 模型正则化
### 模型检验
### 超参数搜索


## 实战

### 泰坦尼克
### IMDB影评
### MNIST手写数字识别
