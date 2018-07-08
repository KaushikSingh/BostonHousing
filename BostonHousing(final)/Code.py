import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import sklearn
sns.set_style("whitegrid")
sns.set_context("poster")
from matplotlib import rcParams
from pandas.plotting import  scatter_matrix
from sklearn.datasets import load_boston
boston=load_boston()
#print(boston.keys())
#print(boston.data.shape)
#print(boston.feature_names)
dataset=pd.DataFrame(boston.data)
dataset.columns=boston.feature_names
#print(dataset.head(10))
#price is present in target
#print(boston.target.shape)
dataset['price']=boston.target
#print(dataset.head(5))
#print(dataset.describe())
Y=dataset['price']
X=dataset.drop('price',axis=1)
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.33,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
'''acc=LinearRegression()
result=acc.fit(X_train,Y_train)
print(result.coef_,result.intercept_)
Y_pred=acc.predict(X_test)
result=acc.score(X_test,Y_test)
print(result*100)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: ")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices: ")
plt.show()
error= sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(error)'''
from sklearn.cluster import KMeans
#test_size=0.33
#seed=6
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
#print(X_train.shape)
kmeans=KMeans(n_clusters=3)
kmeans=kmeans.fit(X_train)
labels=kmeans.predict(X_train)
print(labels)
centers=kmeans.cluster_centers_
print(centers)
print(len(labels))