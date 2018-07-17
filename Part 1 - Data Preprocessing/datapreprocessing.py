# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:57:24 2018

@author: shipra chauhan
"""
#import the libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#import the data set
dataset=pd.read_csv("Data.csv")


#make them in matrix form
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


#for the missing value  import imputer which helps in finding mean 
from sklearn.preprocessing import Imputer


#for the missing value Nan use the strategy mean on axis 0 for columns and axis=1 for rows
imputer=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)

#fitting the imputer values for the matrix x

imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #lebelencoder is used to lebel numbers to different categories of data whereas onehotencoder is used to assign values to different categories as categories may be understood greater or less than one another while performing operations
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0]) #categorical_features is an attribute giving the index value of the column
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder() #as y is an predictive variable so doesnot not need any onehotencoder method as it is automatically understood
y=labelencoder_y.fit_transform(y)


#splitting the data into train and test sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#the above we have divided the whole data set in fourn parts 
#x_train and y_test are the features and predictive value associated with the feature same for y
#x,y in the attributes represents loading the whole data set 
#test_size=0-2 is an idol number means we are dividing 10 feature rows into 8 for training purpose(x_train and y_train) and the remaining two for the testing purpose (y_test and x_test)



#scaling features
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)



#below is the data preprocessing template with code that has to be put before every model 

#import the libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
 
#splitting the data into train and test sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#scaling features
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
sc_y=StandartScaler()
y_trainsc_y.fit_transform(y_train)"""
