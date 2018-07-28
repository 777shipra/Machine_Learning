# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:01:40 2018

@author: shipra chauhan
"""

'''dataset tells 10 versions of the same ad , have the limited budget , find the best version is best result
we are going to show each ad to 10000 users online
one yes will give one reward and no will give 0 reward 
however reinforecement learning will take conderation of the votes from the beginning and 
show the ad with from past record and decide which version of ad to show to the user 
that is why it is called online learning or interactive learning'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
