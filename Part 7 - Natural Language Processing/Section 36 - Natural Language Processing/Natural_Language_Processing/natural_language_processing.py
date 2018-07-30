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
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])# for having only letters in the review a-zA-Z , replacing them with space inorder to stick the words if we remove some signs and numbers , the index 
    review = review.lower()#make all lower case letters in review
    review = review.split()#spiliting the string into list of strings containing each word seperated
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
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