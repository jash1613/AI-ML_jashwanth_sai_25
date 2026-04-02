# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:13:48 2026

@author: mjash
"""

import pandas as pd
import os 
from sklearn import tree
from sklearn import model_selection
os.chdir(r"G:\ml\titanic")
titanic_train=pd.read_csv(r"G:\ml\titanic\train.csv")
titanic_train.shape
titanic_train.info()
titanic_test=pd.read_csv(r"G:\ml\titanic\test.csv")
titanic_test.shape
titanic_test.info()
titanic_test['Survived']=None
titanic_test.info()
titanic=pd.concat([titanic_train,titanic_test])
titanic.shape
titanic.info()
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()
titanic['Title']=titanic['Name'].map(extract_title)
titanic.Age[titanic['Age'].isnull()]=titanic['Age'].mean()
titanic.Fare[titanic['Fare'].isnull()]=titanic['Fare'].mean()
def convert_age(age):
    if(age>=0 and age<=10):
        return 'Child'
    elif(age<=25):
        return 'Young'
    elif(age<=50):
        return 'Middle'
    else:
        return 'Old'
titanic['Age_cat']=titanic['Age'].map(convert_age)
titanic['FamilySize']=titanic['SibSp']+titanic['Parch']+1
def convert_familysize(size):
    if(size==1):
        return 'Single'
    elif(size<=3):
        return 'Small'
    elif(size<=6):
        return 'Medium'
    else:
        return 'Large'
titanic['FamilySize_Cat']=titanic['FamilySize'].map(convert_familysize)
titanic1=pd.get_dummies(titanic,columns=['Sex','Pclass','Embarked','Age_cat','Title','FamilySize_Cat'])
titanic1.shape
titanic1.info()
titanic2=titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1,inplace=False)
titanic2.shape
titanic2.info()
X_train=titanic2[0:titanic_train.shape[0]]
X_train.shape
X_train.info()
y_train=titanic_train['Survived']
dt=tree.DecisionTreeClassifier(random_state=200)
dt_grid={'max_depth':list(range(10,11)),'min_samples_split':list(range(5,8)),'criterion':['gini','entropy']}
param_grid=model_selection.GridSearchCV(dt,dt_grid,cv=5)
param_grid.fit(X_train,y_train)
param_grid.best_score_
param_grid.best_estimator_
param_grid.score(X_train,y_train)
X_test=titanic2[891:]
X_test.shape
X_test.info()
titanic_test['Survived']=param_grid.predict(X_test)
os.getcwd()
titanic_test.to_csv('Submissimon_eda_fe.csv',columns=['PassengerId','Survived'],index=False)
  