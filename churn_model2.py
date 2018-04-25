# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:34:58 2018

@author: binni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_modelling.csv')
#dataset = dataset.dropna(axis=0)
selected_data= ['CreditScore','Tenure','Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = dataset[selected_data]
y = dataset.Exited

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1, random_state = 0)

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
# make pipeline
# Fitting Decision Tree Classification to the Training set
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
pipeline = make_pipeline(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))
pipeline.fit(X_train, y_train)

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
my_plots = plot_partial_dependence(pipeline,       
                                   features=['CreditScore'], # column numbers of plots we want to show
                                   X=X_train,            # raw predictors data.
                                   feature_names=['CreditScore'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis