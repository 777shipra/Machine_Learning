# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
#header =NONE means that there are no titles in the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#dataset is difficult to read so 
#apriori function requires it's value in string 
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
#the file is in the folder itself
from apyori import apriori
#setting the rules on the transaction list
#min length tells the minimum number of arguments you want in the basket 
#support=no of transaction containing the set of i /total no of transactions
#for support we need to look at the products which are bought frequently atleast 2 or 3 times a day
#support and confidence depends on the dataset 
#untill we are satisfied with the output we can try different values to satisfying revenue
#for support -> 3*7/7500 = suppose a product is purchased 3 times a day and purchased are registered over a week 
#so 3*7 / total no of transactions =0.003
#deafult value of the confidence is 0.4

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#apriori model is very experimental
# Visualising the results
#rules in python are already sorted unlike in R because of the relievence criterian in apriori 
results = list(rules)