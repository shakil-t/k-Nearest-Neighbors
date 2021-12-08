# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:49:24 2021

@author: shakil
"""

import pandas as pd

#loading data
beijing=pd.read_csv("Beijing_labeled.csv")
guangzhou=pd.read_csv("Guangzhou_labeled.csv")
shanghai=pd.read_csv("Shanghai_labeled.csv")
shenyang=pd.read_csv("Shenyang_labeled.csv")

#split into train, test, and validation
train=beijing
Xtrain=train[train.columns[:-1]]
Ytrain=train[train.columns[-1]]

validation=shenyang
Xvalidation=validation[validation.columns[:-1]]
Yvalidation=validation[validation.columns[-1]]

test1=guangzhou
test2=shanghai
Xtest1=test1[test1.columns[:-1]]
Ytest1=test1[test1.columns[-1]]
Xtest2=test2[test2.columns[:-1]]
Ytest2=test2[test2.columns[-1]]

#writing our own StandardScaler for data preproccessing
class StandardScaler:
    def __init__(self):
        pass
    
    #first we have to find the categorial features
    def __feature_type(self, x):
        categorial_features=list(x.columns[x.isin([0,1]).all()])
        categorial_index=[]
        numeric_index=[i for i in range(len(x.columns))]
        for feature in categorial_features:
            categorial_index.append(x.columns.get_loc(feature))
        for index in categorial_index:
            numeric_index.remove(index)
        return numeric_index
    
    def fit_transform(self, x):
        numeric_index=self.__feature_type(x)
        x_copy=x.to_numpy()
        for index in numeric_index:
            column=x_copy[:, index]
            mean=np.mean(column)
            std=np.std(column)
            for i in range(0, len(x_copy)):
                x_copy[i][index]=(x_copy[i][index]-mean)/std
        return x_copy

from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np

#writing the model in fit-predict-score pattern
class KNN:
    def __init__(self, k):
        self.k=k
        self.is_fitted=False
        
    #we use Euclidean distance according to the instructions
    def __distance(self, x1, x2):
        return np.linalg.norm(x1-x2)
        
    def fit(self, x, y):
        if self.k>x.shape[0]:
            raise RuntimeError("Error: Invalid K")
            
        if x.shape[0]!=y.shape[0]:
            raise RuntimeError("Error: X and Y do not match")
        self.y=y.values
        self.x=x
        self.is_fitted=True
        return self
        
    
    def predict(self, x):
        if not self.is_fitted:
            raise RuntimeError("Error: Model not fitted")
        
        result=[]
        ind=[]
        distances=[]
        
        for x1 in x:
            for x2 in self.x: 
                distances.append(self.__distance(x1, x2))
            ind.append(np.argsort(distances)[:self.k])
            distances=[]
        for i in ind:
            result.append(Counter(self.y[i]).most_common()[0][0])
        return np.array(result)
    
    def __accuracy(self, predicted, actual):
        correct=0
        for i in range(len(actual)):
          if actual[i]==predicted[i]:
            correct+=1
        return correct/float(len(actual))
    
    def score(self, x, y):
        if not self.is_fitted:
            raise RuntimeError("Error: Model not fitted")
            
        return self.__accuracy(self.predict(x), y)
    
import matplotlib.pyplot as plt
import matplotlib as mpl

#ploting the scores
def diagram(style="fivethirtyeight"):
    mpl.style.use(style)
    fig, ax=plt.subplots(figsize=(9, 9), dpi=150)
    ax.set_title("Validation Scores", color='#7B68EE')
    ax.plot(validation_scores1, color='#DC143C', label="Our kNN")
    ax.plot(validation_scores2, color='#00FF7F', label="Sklearn's kNN")
    ax.legend(fontsize=20)

#put the model into practise first by preproccessing the data
scaler=StandardScaler()
Xtrain=scaler.fit_transform(Xtrain)
Xtest1=scaler.fit_transform(Xtest1)
Xtest2=scaler.fit_transform(Xtest2)
Xvalidation=scaler.fit_transform(Xvalidation)

validation_scores1=[]
for i in range(1, 100):
    knn=KNN(i)
    knn.fit(Xtrain, Ytrain)
    validation_scores1.append(knn.score(Xvalidation, Yvalidation))

#comparing it to the sklearn model
from sklearn.neighbors import KNeighborsClassifier
validation_scores2=[]
for i in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)
    validation_scores2.append(accuracy_score(knn.predict(Xvalidation), Yvalidation))

diagram()

#overall, our model has a higher accuracy score
#after finding the best k by validation data we fit the model with test data
knn=KNN(k=?)
knn.fit(Xtrain, Ytrain)
print("Training accuracy:", knn.score(Xtrain, Ytrain))
print("Testing accuracy on Guangzhuo", knn.score(Xtest1, Ytest1))
print("Testing accuracy on Shanghai", knn.score(Xtest2, Ytest2))
