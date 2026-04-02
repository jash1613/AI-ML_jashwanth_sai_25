# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:19:40 2026

@author: mjash
"""

import os
import pandas as pd
import joblib
titanic_test=pd.read_csv(r"G:\ml\titanic\test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()]=titanic_test['Fare'].mean()
titanic_test1=pd.get_dummies(titanic_test,columns=['Pclass','Sex','Embarked'])
titanic_test1.shape
titanic_test1.info()
x_test=titanic_test1.drop(['PassengerId','Age','Cabin','Ticket','Name'], axis=1, inplace=False)
os.getcwd()
jash=joblib.load("TitanicVer2.pkl")
titanic_test['Survived']=jash.predict(x_test)
titanic_test.to_csv("submissionpickle.csv",columns=['PassengerId','Survived'], index=False)