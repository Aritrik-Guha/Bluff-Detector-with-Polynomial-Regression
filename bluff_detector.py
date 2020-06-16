# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:05:21 2020

@author: KIIT
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
ds=pd.read_csv('Position_salaries.csv')
ds.head()

#Creating x and y variables
x=ds.iloc[:, 1:2].values
y=ds.iloc[:, 2].values

#importing libraries for the detector
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#preparing the model 
pr=PolynomialFeatures(degree=5)
Xp=pr.fit_transform(x)
lr=LinearRegression()
lr.fit(Xp,y)

#for better visualization, we modify the x label separation 
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))

#visualization for detection
plt.scatter(x,y,color='green')
plt.plot(x,lr.predict(pr.fit_transform(x)),color='red')
plt.title('Bluff Detector using Polynomial Regression')
plt.xlabel('Positions at previous company')
plt.ylabel('Salaries of different positions')
plt.show()

#prediction-suppose recruit says he had been a Senior Partner for more than 1 
#year and had a salary of 400k. So, we can expect him to be at a level 7.5.
lr.predict(pr.fit_transform([[7.5]])) 

#prediction says he could earn a salary of 237k , but he said that his salary 
#was 400k. So , its a complete bluff.