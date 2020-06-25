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

### Using Polynomial Regression

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

### Using SVR

#Creating x and y variables
x1=ds.iloc[:, 1:2].values
y1=ds.iloc[:, 2].values
y1 = y1.reshape(len(y1),1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x1=sc_x.fit_transform(x1)
y1=sc_y.fit_transform(y1)


#importing libraries for the detector
from sklearn.svm import SVR

#preparing the model
rg=SVR(kernel='rbf')
rg.fit(x1,y1)

#for better visualization, we modify the x label separation 
x_grid = np.arange(min(sc_x.inverse_transform(x1)), max(sc_x.inverse_transform(x1)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

#visualization for detection
plt.scatter(sc_x.inverse_transform(x1), sc_y.inverse_transform(y1), color = 'green')
plt.plot(x_grid, sc_y.inverse_transform(rg.predict(sc_x.transform(x_grid))), color = 'red')
plt.title('Bluff Detector using SVR')
plt.xlabel('Positions at previous company')
plt.ylabel('Salaries of different positions')
plt.show()

#prediction-suppose recruit says he had been a Senior Partner for more than 1 
#year and had a salary of 400k. So, we can expect him to be at a level 7.5.
sc_y.inverse_transform(rg.predict(sc_x.transform([[7.5]])))
#prediction says he could earn a salary of 263k , but he said that his salary 
#was 400k. So , its a complete bluff.

### Using Decision Tree

#Creating x and y variables
x2=ds.iloc[:, 1:2].values
y2=ds.iloc[:, 2].values

#importing libraries for the detector
from sklearn.tree import DecisionTreeRegressor
rg2=DecisionTreeRegressor(random_state=0)
rg2.fit(x2,y2)

#for better visualization, we modify the x label separation 
x_grid=np.arange(min(x2),max(x2),0.001)
x_grid=x_grid.reshape((len(x_grid),1))

#visualization for detection
plt.scatter(x2,y2,color='green')
plt.plot(x_grid,rg2.predict(x_grid),color='red')
plt.title('Bluff Detector using Decision Tree')
plt.xlabel('Positions at previous company')
plt.ylabel('Salaries of different positions')
plt.show()

#prediction-suppose recruit says he had been a Senior Partner for more than 1 
#year and had a salary of 400k. So, we can expect him to be at a level 7.5.
rg2.predict([[7.5]])
#prediction says he could earn a salary of 200k , but he said that his salary 
#was 400k. So , its a complete bluff.

### Using Random Forest

#Creating x and y variables
x3=ds.iloc[:, 1:2].values
y3=ds.iloc[:, 2].values

#importing libraries for the detector
from sklearn.ensemble import RandomForestRegressor
rg3=RandomForestRegressor(n_estimators=100, random_state=0)
rg3.fit(x3,y3)

#for better visualization, we modify the x label separation 
x_grid=np.arange(min(x3),max(x3),0.001)
x_grid=x_grid.reshape((len(x_grid),1))

#visualization for detection
plt.scatter(x3,y3,color='green')
plt.plot(x_grid,rg3.predict(x_grid),color='red')
plt.title('Bluff Detector using Random Forest')
plt.xlabel('Positions at previous company')
plt.ylabel('Salaries of different positions')
plt.show()

#prediction-suppose recruit says he had been a Senior Partner for more than 1 
#year and had a salary of 400k. So, we can expect him to be at a level 7.5.
rg3.predict([[7.5]])
#prediction says he could earn a salary of 233k , but he said that his salary 
#was 400k. So , its a complete bluff.