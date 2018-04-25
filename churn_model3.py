# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:36:37 2018

@author: binni
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_modelling2.csv')
#dataset = dataset.dropna(axis=0)
selecting = [ 'Age', 'CreditScore', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary']
X = dataset[selecting]
y = dataset.Exited
#dataset['CreditScore'].value_counts().plot('bar')

#dataset.plot(kind = 'scatter', x = 'Exited', y ='CreditScore' )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
pipeline = make_pipeline(LogisticRegression(random_state = 0))
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = pipeline, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

