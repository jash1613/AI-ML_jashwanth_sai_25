# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:07:29 2026

@author: mjash
"""



#EDA, FE, Combine both train+test data and Extract all Titles

import pandas as pd
import os

#from sklearn import preprocessing
#from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn import model_selection

#Change working directory
os.chdir(r"G:\ml\titanic")

titanic_train = pd.read_csv("train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('test.csv')
titanic_test.shape
titanic_test.info()

titanic_test.Survived = None

#Let's excercise by concatinating both train and test data
#Concatenation is Bcoz to have same number of rows and columns so that our job will be easy
titanic = pd.concat([titanic_train, titanic_test])
titanic.shape
titanic.info()

#name = "Allen, Master. William Henry"
#Extract and create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
   
titanic['Title'] = titanic['Name'].map(extract_title)

titanic.Age[titanic['Age'].isnull()] = titanic['Age'].mean()
titanic.Fare[titanic['Fare'].isnull()] = titanic['Fare'].mean()


#creaate categorical age column from age
#It's always a good practice to create functions so that the same can be applied on test data as well
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
#Transform numerical Age column into categorical Age_Cat column
titanic['Age_Cat'] = titanic['Age'].map(convert_age)

#Create a new column FamilySize by combining SibSp and Parch and seee we get any additioanl pattern recognition than individual
#Add +1 for including passenger him self to Family Size
titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
#Convert numerical FamilySize column to categorical FamilySize_Cat column
titanic['FamilySize_Cat'] = titanic['FamilySize'].map(convert_familysize)
titanic['Cabin']=titanic['Cabin'].str[0]
#Now we got 3 new columns, Title, Age_Cat, FamilySize_Cat
#convert categorical columns to one-hot encoded columns including  newly created 3 categorical columns
#There is no other choice to convert categorical columns to get_dummies in Python
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Cabin','Embarked', 'Age_Cat', 'Title', 'FamilySize_Cat'])
titanic1.shape
titanic1.info()

#Drop un-wanted columns for faster execution and create new set called titanic2
titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Survived'], axis=1, inplace=False)
#See how many columns are there after 3 additional columns, one hot encoding and dropping
titanic2.shape 
titanic2.info()
#Splitting train and test data
X_train = titanic2[0:891] #0 t0 891 records
X_train.shape
X_train.info()
X_train.head(10)
y_train = titanic_train['Survived']

#Let's build the model
#If we don't use random_state parameter, system can pick different values each time and we may get slight difference in accuracy each time you run.
dt = tree.DecisionTreeClassifier(random_state = 1)

#Add parameters for tuning
dt_params = {'max_depth':[30, 40], 'min_samples_split':[9, 15], 'criterion':['gini', 'entropy']}
TitanicModel = model_selection.GridSearchCV(dt, dt_params, cv=10) #Evolution of tree
TitanicModel.fit(X_train, y_train) #Building the tree
TitanicModel.best_score_
TitanicModel.best_params_
TitanicModel.best_estimator_

#Now let's predict on test data
X_test = titanic2[891:]
X_test.shape
X_test.info()
#Predict on Test data
titanic_test['Survived'] = TitanicModel.predict(X_test)
os.chdir(r"G:\ml\titanic")
from matplotlib import pyplot as pt
from sklearn.tree import plot_tree
pt.figure(figsize=(200, 200))
plot_tree(TitanicModel.best_estimator_, feature_names=X_train.columns)
pt.savefig(r"G:\ml\titanic\eda_fe.pdf", format='pdf')

os.getcwd()
titanic_test.to_csv('DT_EDA_FE_CV_Optimize.csv', columns=['PassengerId','Survived'], index=False)