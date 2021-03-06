# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000 #total no of users
d = 10#total no of ads
ads_selected = []#record of the ad selected according to the max upper bound
numbers_of_selections = [0] * d # 0 because initially no ad is selected 
sums_of_rewards = [0] * d#initially 0
total_reward = 0
for n in range(0, N):#for all the users
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):#for allthe ads
        if (numbers_of_selections[i] > 0):
            #formula for UCB
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]#real reward n index of row and ad column 
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward #increamenting the sum of reward of particular ad
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)#histogram
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()