# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#delimiter because we chose tab seperated values because ' 
# quoting = 3 value is not ignore the " 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
''' getting rid of the texts which are not usefull in prediction such as on the is 
also the punctuations (...)
applying stamin means (loved is the form of love ) so we will only take love not loved from the reviews 
stamming is applied in order not to have too many words in the end without changing their meaning in the reviews
getting rid of the capitals 
and then we will move forward to bag of words 
bag of words which is the tokenization method that is it will split the reviews , give one word and will create a sparx matrix '''
import re #used to clean the text
import nltk
nltk.download('stopwords')#it is the list containing the word which we do not want in the review
from nltk.corpus import stopwords#to import stopwords to spyder
from nltk.stem.porter import PorterStemmer#for importing stemming
corpus = []#list containing 1000 clean reviews 
#for all the 1000 reviews in the dataset
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])# for having only letters in the review a-zA-Z , replacing them with space inorder to stick the words if we remove some signs and numbers , the index 
    review = review.lower()#make all lower case letters in review
    review = review.split()#spiliting the string into list of strings containing each word seperated
    ps = PorterStemmer()#creating an object before appling stemming on the review
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]#applying stemming and stopwards in english on the reviews 
    review = ' '.join(review) #joining back the list of strings and seperate them with space
    corpus.append(review)
'''


Sparsity is a property of our data which we cannot change. We can use our knowledge about it to apply methods which take it into account and as such more efficient on sparse data.

A corpus is a object in our programming environment for containing text data. It is convenient to keep it in a compact form and to retrieve what we want quickly.

'''
# Creating the Bag of Words model
''' the bag of words will help us in giving one word to each review (unique) and making it a column 
that is why we cleaned the reviews before applying this 
a table will be created containing
rows - corpus
columns will be unique words
that will create a sparx matrix through the process of tokenization leading to scarcity and in ML we try to reduce sparcity as much as possible
why ?
-to predict positive or negative by training the machine  
here we are doing nothing as classification  '''
from sklearn.feature_extraction.text import CountVectorizer #for tokenization as manually it will take alot of time 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()#this step will create the sparx matrix , X is the matrix of features
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)