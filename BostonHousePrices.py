# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:21:36 2019

@author: Naveena
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston=load_boston()
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#Fitting simple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

print(regressor.coef_)
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(regressor.score(X_test, y_test))) 

# plotting residual errors in training data 
plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data') 
  
# plotting residual errors in test data 
plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show()

# =============================================================================
# #Visualizing the Test set results
# plt.scatter(X_test,y_test,color='red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title("Features Vs Price (Test set)")
# plt.xlabel("Features")
# plt.ylabel("Price")
# plt.show()
# =============================================================================
