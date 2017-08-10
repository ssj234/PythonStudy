# Python机器学习及实践

## 主要流程

1. 分析训练数据和测试数据，主要是对数据的分布和特征意义进行总结和处理。如选取有意义的特征，填充缺失值，正则化，归一化，二次项等操作，还可以使用绘图工具查看数据分布情况。

* 查看数据分布：

```
y_train.value_counts() # 查看每个数值的数量
y_train.info() # DataFrame
y_train.describe() # 查看数据数据、均值、最大、最小值等
```

* 拆分数据集合：

```
from sklearn.cross_validation import train_test_split
# 第0列为列标，1-9为X数据 10为Y的值
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
```

* 填充数据： 有时数据的某些列会有缺失，需要对数据进行填充或丢弃,填充时可以使用平均值或众数
```
X['age'].fillna(X['age'].mean(),inplace=True) # 对缺失的年龄列填充平均值
```

* 标准化数据:保证每个维度的特征数据方差为1，均值为0，使得预测结果不被某些维度过大的特征值而主导,公式为：(X-mean)/std，减去平均值后除以方差

```
from sklearn.preprocessing import StandardScaler
#标准化数据，
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
```

* 文本特征向量
```
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
```

* 文本向量化
```
例如，对如下的DataFrame进行向量化
    A   B   C   D
0  a1  b1  c1  d1
1  a1  b2  c2  d2
2  a1  b3  c3  d2
3  a1  b4  c3  d2
>>> vec = DictVectorizer(sparse=False) # 初始化 非稀疏矩阵
>>> x = df.to_dict(orient='record') # 将数据转为字典
[{'A': 'a1', 'C': 'c1', 'B': 'b1', 'D': 'd1'}, {'A': 'a1', 'C': 'c2', 'B': 'b2', 'D': 'd2'}, {'A': 'a1', 'C': 'c3', 'B': 'b3', 'D': 'd2'}, {'A': 'a1', 'C': 'c3', 'B': 'b4', 'D': 'd2'}]
>>> vec.fit_transform(x) # 进行向量化
>>> vec.feature_names_  # 特征名称
['A=a1', 'B=b1', 'B=b2', 'B=b3', 'B=b4', 'C=c1', 'C=c2', 'C=c3', 'D=d1', 'D=d2']
>>> vec.fit_transform(x)  # 结果
array([[ 1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
       [ 1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.],
       [ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.]])

```

对性别进行处理时，可以直接修改其值
```
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
```

* 降维
```
from sklearn.decomposition import PCA
estimator=PCA(n_components=2) # 将64维压缩到2维
x_pca=estimator.fit_transform(x_digits)
pca_X_train = estimator.fit_transform(X_train)
pca_X_test = estimator.transform(X_test)
```

* 特征选择：其目的是观察每个特征对结果的影响程度，方法是：每次进行又放回的采样，对于某列特征，如titanic中的age维度，第一次观察错误率err1，第二次混淆age，随机添加噪音值，再次观察错误率err2，如果错误率变化明显，说明age对结果的影响较大。

```
# 导入SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif,k=2)
selector.fit(titanic[predictions],titanic['survived'])
scores = -np.log10(selector.pvalues_)
# 展示图像
plt.bar(range(len(predictions)),scores)
plt.xticks(range(len(predictions)),predictions,rotation='vertical')
plt.show()
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
# P(y|x) = P(x|y) * P(y) / P(x)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)

# 决策树
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)

# 梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)
```

回归工具：
```
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

# svm
from sklearn.svm import SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)


rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

# KNN算法
from sklearn.neighbors import KNeighborsRegressor
## 初始化K近邻回归器，预测方式为平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

## 初始化K近邻回归器，预测方式为距离加权回归
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# 决策树回归
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict = dtr.predict(X_test)

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
```

聚类：
```
from sklearn.cluster import KMeans

## 初始化KMeans模型并设置聚类中心数量为10
kmeans=KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

# 度量
from sklearn import metrics
print metrics.adjusted_rand_score(Y_test,y_pred)
# 轮廓系数
sc_socre=silhouette_score(X,kmeans_model.labels_,metric='euclidean')
```


* 交叉验证： 主要思路是将所有的样本拆分为两个部分：训练集和测试集。 将训练集分为三份，Train1,Train2,Train3；对这三份进行交叉验证，第一次使用1和2进行训练，使用3进行测试；第二次使用2和3进行训练，使用1进行测试；最后，使用1和3进行训练，使用2进行验证。得到了三个准确率，最后进行平均化。

```
from sklearn.cross_validation import KFold
kf = KFold(titanic.shape[0],n_folds=3,random_state=1) # 3层交叉验证
predictions = ['pclass','sex','age','embarked','namelength','homelength']
alg = LinearRegression() # 线性回归
for train,test in kf:
    train_predictions = titanic[predictions].iloc[train,:]
    train_target = titanic['survived'].iloc[train,:]
    alg.fit(train_predictions,train_target) # 训练
    test_predictions = alg.predict(titanic[predictions].iloc[train,:]) # 预测
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
* MAE:平均绝对误差  MAE = |预测值-真实值|绝对值之和 / m
* MSE：均方误差  MSE = |预测值-真实值|绝对值的平方 / m
* R-squard  1 - (测试数据真实值Y的方差)/(回归值和真实值之间的平方差异)
* R-squard用来衡量回归结果的波动可被真实值验证的百分比
```
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

print 'LinearRegression R-squared',r2_score(y_test,lr_y_predict)
print 'LinearRegression MSE ',mean_squared_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(lr_y_predict))
print 'LinearRegression MAE ',mean_absolute_error(ss_Y.inverse_transform(y_test),ss_Y.inverse_transform(lr_y_predict))

```


## 实用技巧

### 特征提升
### 模型正则化
### 模型检验
### 超参数搜索


## 实战

### 泰坦尼克
### IMDB影评
### MNIST手写数字识别
