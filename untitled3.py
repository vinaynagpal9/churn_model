# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:38:16 2018

@author: binni
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Churn_modelling.csv')
#dataset = dataset.dropna(axis=0)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#dataset = drop.dataset['EstimatedSalary'])

dataset['CreditScore'].value_counts().plot('bar')

alpha_color = 0.5
dataset['Exited'].value_counts().plot(kind = 'bar')

alpha_color = 0.7
dataset['IsActiveMember'].value_counts().plot(kind = 'bar')

alpha_color = 0.5
dataset['HasCrCard'].value_counts().plot(kind = 'bar')

alpha_color = 0.5
dataset['NumOfProducts'].value_counts().plot(kind = 'bar')

alpha_color = 0.5
dataset['Gender'].value_counts().plot(kind = 'bar')

alpha_color = 0.5
dataset['Geography'].value_counts().plot(kind = 'bar')

dataset.plot(kind = 'scatter', x = 'Exited', y ='Age' )

dataset.plot(kind = 'scatter', x = 'Exited', y ='EstimatedSalary' )

dataset.plot(kind = 'scatter', x = 'Exited', y ='Balance' )

dataset.plot(kind = 'scatter', x = 'Exited', y ='CreditScore' )

dataset.plot(kind = 'scatter', x = 'Exited', y ='IsActiveMember' )

bins = [0, 10, 20, 30 ,40, 50, 60, 70, 80]
dataset['agebin']= pd.cut(dataset['Age'], bins)
dataset[dataset['Exited'] == 1]['agebin'].value_counts().sort_index().plot(kind = 'bar')

