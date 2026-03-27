# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:31:02 2026

@author: mjash
"""

import pandas as pd
from sklearn import tree
titanic_train=pd.read_csv(r"G:\ml\titanic\train.csv")
titanic_train.shape
titanic_train.info()
titanic_train.describe()
x_titanic_train=titanic_train[['Pclass','SibSp','Parch']]
y_titanice_train=titanic_train['Survived']
dt=tree.DecisionTreeClassifier()
dt.fit(x_titanic_train,y_titanice_train)
titanic_test=pd.read_csv(r"G:\ml\titanic\test.csv")
x_test=titanic_test[['Pclass','Sibsp','Parch']]
titanic_test['Survived']=dt.predict(x_test)
import os
os.getcwd()
titanic_test.to_csv("Attempt1.csv",columns=['PassengerId','Survived'], index=False)