# Artificial Neural Network


# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
''' it performs numerical computation very fast both on CPU and  GPU(process of graphic purposes)
 '''

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
#same , by google , these two are used for deep learning  
 
 
# Installing Keras
# pip install --upgrade keras
 #it wraps the above two , to build models , in only view lines of code

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #required to initial our neural network
from keras.layers import Dense#to create the neural network 

# Initialising the ANN
classifier = Sequential() #two ways of definning one by sequential or by definning a graph 

# Adding the input layer and the first hidden layer
#output_dim= no of nodes in the layer +nodes in output layer/2 =11+1/2=6
#how many nodes ?- no rule -experimenting with k-4 cross validation 
#init- initialise the weights 
#activation function is the rectifier function (relu)for input layer and sigmoid for the output layer
#input_dim= it is compulsory argument -> as we are initialising our nn , so not initialising for the next hidden rate 

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
#as we need probabilities we will use sigmoid function 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
#optimizer - is the algo used to find the optimal set of weights in nn , best weights 
#stachastic GD is we are going to use and one type of SGD is adam 
#loss-> adam is based on a loss function (cos function ) and as our activation function is sigmoid 
#when the output is in binary we use binary _crossentropy and if more than categorical_crossentropy
#metrics-> it improves the performance of the models after weights are updated the algo uses this function
#metric expects a list and we will add only one attribute
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#batch_size->tells after which u want to update the weights 
#nb_epoch->happens after the whole ANN
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)