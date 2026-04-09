# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:56:51 2026

@author: mjash
"""

import os
from sklearn import ensemble
from sklearn import model_selection
import pandas as pd

os.chdir(r"G:\ml\titanic")

titanic_train=pd.read_csv(r"G:/ml/titanic/train.csv")
titanic_train.shape
titanic_train.info()

#one hot encoding..
titanic_train1=pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(5)

#dropping usless columnss
x_train=titanic_train1.drop(['PassengerI','Age','Cabin','Ticket','Name','Survived'],axis=1,inplace=False)
y_train=titanic_train1['Survived']


#randomforest classfier
rf=ensemble.RandomForestClassifier(random_state=1)
#n_estimators:no of trees to be built
#max_features no features to try with 

#declare hyperparamters
rf_grid={'n_estimators':list(range(250,551,50)),'max_features':[11],'criterion':['entropy','gini']}
rf1=model_selection.GridSearchCV(rf, rf_grid,cv=10,n_jobs=10)
rf1.fit(x_train,y_train)

#knowning best parameters and score 
rf1.best_estimator_
rf1.best_score_
rf1.score(x_train,y_train)

#same for test data
titanic_test=pd.read_csv(r"G:\ml\titanic\test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()]=titanic_test['Fare'].mean()

#one hot encoding for test data
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test=titanic_test1.drop(['PassengerId','Age','Cabin','Ticket','Name'], axis=1,inplace=False)
titanic_test['Survived']=rf1.predict(X_test)
titanic_test.to_csv("Submission_rf.csv",columns=['PassengerId','Survived'],index=False)
