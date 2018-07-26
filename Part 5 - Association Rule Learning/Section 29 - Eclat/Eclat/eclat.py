# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 20:09:19 2018

@author: shipra chauhan
"""

# Eclat using the apyori.py file
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([x for x in dataset.values[i] if str(x) != 'nan'])
 
# Training Apriori on the dataset (with no min confidence and lift, for Eclat)
from apyori import apriori
rules = apriori(transactions, min_support = 0.004, min_confidence = 0, min_lift = 0)
 
# Visualising the results
results = list(rules)
 
# Sort by the support in decending order
results.sort(key=lambda tup: tup[1], reverse=True)
 
# set the min length for the results
min_length = 2
 
# Have the results be visible (based on Q&A examples)
results_list = []
for i in range(0, len(results)):
    if len(results[i][0]) >= min_length:
        results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
