#!/usr/bin/env python
# coding: utf-8

# Exercise 2 - Simple Linear Regression
# =====================
# 
# We want to know how to make our chocolate-bar customers happier. To do this, we need to know which chocolate bar _features_ predict customer happiness. For example, customers may be happier when chocolate bars are bigger, or when they contain more cocoa. 
# 
# We have data on customer happiness when eating chocolate bars with different features. Lets look at the relationship between happiness and bar size.
# 
# Step 1
# --
# 
# First, lets have a look at our data.
# 
# #### In the cell below replace the text `<printDataHere>` with `print(dataset.head())` and then press __Run__ in the toolbar above (or press __Shift+Enter__).

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as graph
import statsmodels.formula.api as smf
from scipy import stats

dataset = pd.read_csv('Data/chocolate data.txt', index_col=False, sep="\t",header=0)
    
###
# REPLACE <PrintDataHere> WITH print(dataset.head())
###
print(dataset.head())
###


# The data represents 100 different variations of chocolate bars and the measured customer happiness for each one. 
# 
# Step 2
# --
# 
# We want to know which chocolate bar features make customers happy.
# 
# The example below shows a linear regression between __cocoa percentage__ and __happiness__. You can read through the comments to understand what is happening. 
# 
# #### __Run the code__ to to see the output visualized.

# In[4]:


# Run this cell!

# DO NOT EDIT ANY OF THIS CODE

# Define a function to perform a linear regression
def PerformLinearRegression(formula):

    # This performs linear regression
    lm = smf.ols(formula = formula, data = dataset).fit()

    featureName=formula.split(" ")[-1]
    
    # get the data for the x parameter (our feature)
    train_X=dataset[featureName]
    
    # This makes and shows a graph
    intercept=lm.params[0]
    slope=lm.params[1]
    line = slope * train_X + intercept
    graph.plot(train_X, line, '-', c = 'red')
    graph.scatter(train_X, dataset.customer_happiness)
    graph.ylabel('customer_happiness')
    graph.xlabel(featureName)
    graph.show()

# This performs the linear regression steps listed above
# The text in red is the formula for our regression
PerformLinearRegression('customer_happiness ~ cocoa_percent')


# In the scatter plot above, each point represents an observation for a single chocolate bar.
# 
# It seems that __more cocoa makes customers more happy__. We can tell, because as we increase the amount of cocoa (x axis) the amount of customer happiness (y axis) increases. 
# 
# Step 3
# ------
# 
# Let's look at some other features.
# 
# #### Below, replace the text `<addFeatureHere>` with __`weight`__ to see if heavier chocolate bars make people happier.
# 
# Also try the variables `sugar_percent` and  `milk_percent` to see if these improve customers' experiences.

# In[5]:


###
# CHANGE cocoa_percent TO weight IN THE LINE BELOW
###
PerformLinearRegression('customer_happiness ~ weight')
###


# In[6]:


PerformLinearRegression('customer_happiness ~ sugar_percent')
###


# In[7]:


PerformLinearRegression('customer_happiness ~ milk_percent')
###


# It looks like heavier chocolate bars make customers happier. The amount of milk or sugar, however, don't seem to make customers happier. 
# 
# Conclusion
# ---
# You have run a simple linear regression. This told us that if we want to make a chocolate bar that will make customers happy, it should be large and contain a lot of cocoa.
# 
# Well done! You can now go back to the course and click __'Next Step'__ to move onto using linear regression with multiple features.
