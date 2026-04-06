# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:21:31 2026

@author: mjash
"""

import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#training data 
titanic_train=pd.read_csv(r"G:\ml\titanic\train.csv")
os.getcwd()
os.chdir(r"G:\ml\titanic")
titanic_train.shape
titanic_train.info()

#one hot ending transforming 
titanic_train1=pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(5)

#dropping unwanted columnss
X_train=titanic_train1.drop(['PassengerId','Age','Cabin','Ticket','Name','Survived'], axis=1, inplace=False)
y_train=titanic_train['Survived']

#loading decision treee
dt=tree.DecisionTreeClassifier()

#baggingclasfior=divding tree into 5 with same columns
bag_tree=ensemble.BaggingClassifier(estimator=dt,n_estimators=5)
scores=model_selection.cross_val_score(bag_tree,X_train,y_train,cv=2)
bag_tree.fit(X_train,y_train)

bag_tree.score(X_train,y_train)
print(scores)
print(scores.mean())

#bagging with hyperparemeters and using gridsearch cv

bag_tree2=ensemble.BaggingClassifier(estimator=dt,n_estimators=6,random_state=5)
bag_grid={'criterion':['entropy','gini']}

bag_grid_estimator=model_selection.GridSearchCV(bag_tree2,bag_grid,n_jobs=6)
bag_tree2.fit(X_train,y_train)
print(os.getcwd())
os.chdir(r"G:\ml\titanic")
#printing bagging trees
i=0
for model in bag_tree2.estimators_:
    plt.figure(figsize=(200,200))
    plot_tree(model,feature_names=X_train.columns)
    plt.savefig("Bagging_tree"+str((i))+".pdf", format="pdf")
    plt.close()
    i=i+1
    