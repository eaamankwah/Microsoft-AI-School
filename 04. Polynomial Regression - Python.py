#!/usr/bin/env python
# coding: utf-8

# Exercise 4 - Polynomial Regression
# ========
# 
# Sometimes our data doesn't have a linear relationship, but we still want to predict an outcome.
# 
# Suppose we want to predict how satisfied people might be with a piece of fruit, we would expect satisfaction would be low if the fruit was under ripened or over ripened. Satisfaction would be high in between underripened and overripened.
# 
# This is not something linear regression will help us with, so we can turn to polynomial regression to help us make predictions for these more complex non-linear relationships!

# Step 1
# ------
# 
# In this exercise we will look at a dataset analysing internet traffic over the course of the day. Observations were made every hour over the course of several days. Suppose we want to predict the level of traffic we might see at any time during the day, how might we do this?
# 
# Let's start by opening up our data and having a look at it.
# 
# #### In the cell below replace the text `<printDataHere>` with `print(dataset.head())`, and __run the code__ to see the data.

# In[ ]:


# This sets up the graphing configuration
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as graph
get_ipython().run_line_magic('matplotlib', 'inline')
graph.rcParams['figure.figsize'] = (15,5)
graph.rcParams["font.family"] = "DejaVu Sans"
graph.rcParams["font.size"] = "12"
graph.rcParams['image.cmap'] = 'rainbow'
graph.rcParams['axes.facecolor'] = 'white'
graph.rcParams['figure.facecolor'] = 'white'
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data/traffic_by_hour.csv')

###
# BELOW, REPLACE <printDataHere> WITH print(dataset.head()) TO PREVIEW THE DATASET ---###
###
<printDataHere>
###


# Step 2
# -----
# 
# Next we're going to flip the data with the transpose method - our rows will become columns and our columns will become rows. Transpose is commonly used to reshape data so we can use it. Let's try it out.
# 
# #### In the cell below find the text `<addCallToTranspose>` and replace it with `transpose`

# In[ ]:


### 
# REPLACE THE <addCallToTranspose> BELOW WITH transpose
###
dataset_T = np.<addCallToTranspose>(data)
###

print(dataset_T)


# Now lets visualize the data. 
# 
# #### Replace the text `<addSampleHere>` with `sample` and then __run the code__.

# In[ ]:


# Let's visualise the data!

###
# REPLACE <addSampleHere> BELOW WITH sample
###
for sample in range(0, dataset_T.shape[1]):
    graph.plot(data.columns.values, data_t[sample])
###

graph.xlabel('Time of day')
graph.ylabel('Internet traffic (Gbps)')
graph.show()


# Step 3
# -----
# 
# This all looks a bit busy, let's see if we can draw out a clearer pattern by taking the __average values__ for each hour.
# 
# #### In the cell below find all occurances of `<replaceWithHour>` and replace them with `hour` and then __run the code__.

# In[ ]:


# We want to look at the mean values for each hour.

hours = dataset.columns.values

###
# REPLACE THE <replaceWithHour>'s BELOW WITH hour
###
train_Y = [dataset[<replaceWithHour>].mean() for <replaceWithHour> in hours]  # This will be our outcome we measure (label) - amount of internet traffic
train_X = np.transpose([int(<replaceWithHour>) for <replaceWithHour> in hours]) # This is our feature - time of day
###

# This makes our graph, don't edit!
graph.scatter(traing_X, train_Y)
for sample in range(0,dataset_T.shape[1]):
    graph.plot(hours, dataset_T[sample], alpha=0.25)
graph.xlabel('Time of day')
graph.ylabel('Internet traffic (Gbps)')
graph.show()


# This alone could help us make a prediction if we wanted to know the expected traffic exactly on the hour.
# 
# But, we'll need to be a bit more clever if we want to make a __good__ prediction of times in between.

# Step 4
# ------
# 
# Let's use the midpoints in between the hours to analyse the relationship between the __time of day__ and the __amount of internet traffic__.
# 
# Numpy's `polyfit(x,y,d)` function allows us to do polynomial regression, or more precisely least squares polynomial fit.
# 
# We specify a __feature $x$ (time of day)__, our __label $y$ (the amount of traffic)__, and the __degree $d$ of the polynomial (how curvy the line is)__.
# 
# #### In the cell below find the text `<replaceWithDegree>`, replace it with the value `1` then __run the code__.

# In[ ]:


# Polynomials of degree 1 are linear!
# Lets include this one just for comparison

###
# REPLACE THE <replaceWithDegree> BELOW WITH 1
###
poly_1 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
###


# Let's also compare a few higher-degree polynomials.
# 
# #### Replace the `<replaceWithDegree>`'s below with numbers, as directed in the comments.

# In[ ]:


###
# REPLACE THE <replaceWithDegree>'s BELOW WITH 2, 3, AND THEN 4
###
poly_2 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
poly_3 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
poly_4 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
###

# Let's plot it!
graph.scatter(train_X, train_Y)
xp = np.linspace(0, 24, 100)

# black dashed linear degree 1
graph.plot(xp, np.polyval(poly_1, xp), 'k--')
# red degree 2
graph.plot(xp, np.polyval(poly_2, xp), 'r-')
# blue degree 3
graph.plot(xp, np.polyval(poly_3, xp), 'b-') 
# yellow degree 4
graph.plot(xp, np.polyval(poly_4, xp), 'y-') 

graph.xticks(train_X, data.columns.values)
graph.xlabel('Time of day')
graph.ylabel('Internet traffic (Gbps)')
graph.show()


# None of these polynomials do a great job of generalising the data. Let's try a few more.
# 
# #### Follow the instructions in the comments to replace the `<replaceWithDegree>`'s and then __run the code__.

# In[ ]:


###
# REPLACE THE <replaceWithDegree>'s 5, 6, AND 7
###
poly_5 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
poly_6 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
poly_7 = np.polyfit(train_X, train_Y, <replaceWithDegree>)
###

# Let's plot it!
graph.scatter(train_X, train_Y)
xp = np.linspace(0, 24, 100)

# black dashed linear degree 1
graph.plot(xp, np.polyval(poly_1, xp), 'k--')
# red degree 5
graph.plot(xp, np.polyval(poly_5, xp), 'r-') 
# blue degree 6
graph.plot(xp, np.polyval(poly_6, xp), 'b-') 
# yellow degree 7
graph.plot(xp, np.polyval(poly_7, xp), 'y-') 

graph.xticks(train_X, data.columns.values)
graph.xlabel('Time of day')
graph.ylabel('Internet traffic (Gbps)')
graph.show()


# It looks like the 5th and 6th degree polynomials have an identical curve. This looks like a good curve to use.
# 
# We could perhaps use an even higher degree polynomial to fit it even more tightly, but we don't want to overfit the curve, since we want just a generalisation of the relationship.
# 
# Let's see how our degree 6 polynomial compares to the real data.
# 
# #### Replace the text `<replaceWithPoly6>` with `poly_6` and __run the code__.

# In[ ]:


for row in range(0,data_t.shape[1]):
    graph.plot(data.columns.values, data_t[row], alpha = 0.5)

###
# REPLACE <replaceWithPoly6> BELOW WITH poly_6 - THE POLYNOMIAL WE WISH TO VISUALIZE
###    
graph.plot(xp, np.polyval(<replaceWithPoly6>, xp), 'k-')
###

graph.xlabel('Time of day')
graph.ylabel('Internet traffic (Gbps)')
graph.show()


# Step 5
# ------
# 
# Now let's try using this model to make a prediction for a time between 00 and 24.
# 
# #### In the cell below follow the instructions in the code to replace `<replaceWithTime>` and `<replaceWithPoly6>` then __run the code__.

# In[ ]:


###
# REPLACE <replaceWithTime> BELOW WITH 12.5 (this represents the time 12:30)
###
time = 12.5
###

###
# REPLACE <replaceWithPoly6> BELOW WITH poly_6 SO WE CAN VISUALIZE THE 6TH DEGREE POLYNOMIAL MODEL
###
pred = np.polyval(<replaceWithPoly6>, time)
###

print("at t=%s, predicted internet traffic is %s Gbps"%(time,pred))

# Now let's visualise it
graph.plot(xp, np.polyval(<replaceWithPoly6>, xp), 'y-')

graph.plot(time, pred, 'ko') # result point
graph.plot(np.linspace(0, time, 2), np.full([2], pred), dashes=[6, 3], color='black') # dashed lines (to y-axis)
graph.plot(np.full([2], time), np.linspace(0, pred, 2), dashes=[6, 3], color='black') # dashed lines (to x-axis)

graph.xticks(train_X, data.columns.values)
graph.ylim(0, 60)
graph.title('expected traffic throughout the day')
graph.xlabel('time of day')
graph.ylabel('internet traffic (Gbps)')

graph.show()


# Conclusion
# -----
# 
# And there we have it! You have made a polynomial regression model and used it for analysis! This models gives us a prediction for the level of internet traffic we should expect to see at any given time of day.
# 
# You can go back to the course and either click __'Next Step'__ to start an optional step with tips on how to better work with AI models, or you can go to the next module where instead of predicting numbers we predict categories.
