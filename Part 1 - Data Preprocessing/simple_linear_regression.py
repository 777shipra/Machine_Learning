# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:10:55 2018

@author: shipra chauhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
 
#splitting the data into train and test sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#scaling features
#for linear regression we wont do it the library will automatically take care of it 
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
sc_y=StandartScaler()
y_trainsc_y.fit_transform(y_train)"""


#fitting the simple linear regression on the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train) #here we made a machine regressor which is an object and telling it to learn our training data

#predicting the test set results
y_pred=regressor.predict(x_test)

#visualising or plotting the results using matplotlib.pyplot of the training set
#we would be using scatter representation on the graph
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("salary v/s experience of training set",)
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


#visualising or plotting the results using matplotlib.pyplot of the test set
#we would be using scatter representation on the graph
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")#even if we write x_test here are result would be the same as the machine trained(regressor.fit(x_train,y_train))in this equation above gives the same equation for predicting hence the line draw would follow the same equation for x_train and as well as x_test
plt.title("salary v/s experience of test set",)
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()





















