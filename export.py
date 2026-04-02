# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:39:11 2026

@author: mjash
"""

import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import joblib
os.getcwd()
titanic_train=pd.read_csv(r"G:\ml\titanic\train.csv")
titanic_train.shape
titanic_train.info()
titanic_train1=pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)
x_train=titanic_train1.drop(['PassengerId','Age','Cabin','Ticket','Name','Survived'], axis=1,inplace=False)
y_train=titanic_train1['Survived']
dt=tree.DecisionTreeClassifier(random_state=1)
dt_grid={'max_depth':list(range(10,11)),'min_samples_split':list(range(5,8)),'criterion':['gini','entropy']}
param_grid=model_selection.GridSearchCV(dt, dt_grid,cv=5)
param_grid.fit(x_train, y_train)
print(param_grid.best_score_)
print(param_grid.best_params_)
print(param_grid.score(x_train,y_train))
os.getcwd()
joblib.dump(param_grid,"TitanicVer2.pkl")
