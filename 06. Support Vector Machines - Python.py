#!/usr/bin/env python
# coding: utf-8

# Exercise 6 - Support Vector Machines
# =====
# 
# Support vector machines (SVMs) let us predict categories. This exercise will demonstrate a simple support vector machine that can predict a category from a small number of features. 
# 
# Our problem is that we want to be able to categorise which type of tree an new specimen belongs to. To do this, we will use features of three different types of trees to train an SVM. 
# 
# __Run the code__ in the cell below.

# In[ ]:


# Run this code!
# It sets up the graphing configuration.
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as graph
get_ipython().run_line_magic('matplotlib', 'inline')
graph.rcParams['figure.figsize'] = (15,5)
graph.rcParams["font.family"] = 'DejaVu Sans'
graph.rcParams["font.size"] = '12'
graph.rcParams['image.cmap'] = 'rainbow'


# Step 1
# -----
# 
# First, we will take a look at the raw data first to see what features we have.
# 
# #### Replace `<printDataHere>` with `print(dataset.head())` and then __run the code__.

# In[ ]:


import pandas as pd
import numpy as np

# Loads the SVM library
from sklearn import svm

# Loads the dataset
dataset = pd.read_csv('Data/trees.csv')

###
# REPLACE <printDataHere> with print(dataset.head()) TO PREVIEW THE DATASET
###
<printDataHere>
###


# It looks like we have _four features_ (leaf_width, leaf_length, trunk_girth, trunk_height) and _one label_ (tree_type).
# 
# Let's plot it.
# 
# __Run the code__ in the cell below.

# In[ ]:


# Run this code to plot the leaf features

# This extracts the features. drop() deletes the column we state (tree_type), leaving on the features
allFeatures = dataset.drop(['tree_type'], axis = 1)

# This keeps only the column we state (tree_type), leaving only our label
labels = np.array(dataset['tree_type'])

#Plots the graph
X=allFeatures['leaf_width']
Y=allFeatures['leaf_length']
color=labels
graph.scatter(X, Y, c = color)
graph.title('classification plot for leaf features')
graph.xlabel('leaf width')
graph.ylabel('leaf length')
graph.legend()
graph.show()


# __Run the code__ in the cell below to plot the trunk features

# In[ ]:


# Run this code to plot the trunk features
graph.scatter(allFeatures['trunk_girth'], allFeatures['trunk_height'], c = labels)
graph.title('Classification plot for trunk features')
graph.xlabel('trunk girth')
graph.ylabel('trunk height')
graph.show()


# Step 2
# -----
# 
# Lets make a support vector machine.
# 
# The syntax for a support vector machine is as follows:
# 
# __`model = svm.SVC().fit(features, labels)`__
# 
# Your features set will be called __`train_X`__ and your labels set will be called __`train_Y`__
# 
# #### Let's first run the SVM in the cell below using the first two features, the leaf features.

# In[ ]:


# Sets up the feature and target sets for leaf features

# Feature 1
feature_one = allFeatures['leaf_width'].values

# Feature 2
feature_two = allFeatures['leaf_length'].values

# Features
train_X = np.asarray([feature_one, feature_two]).transpose()

# Labels
train_Y = labels 

# Fits the SVM model
###--- REPLACE THE <makeSVM> WITH THE CODE TO MAKE A SVM MODEL AS ABOVE ---###
model = <makeSVM>
###
print("Model ready. Now plot it to see the result.")


# #### Let's plot it! Run the cell below to visualise the SVM with our dataset.

# In[ ]:


# Run this to plots the SVM model
X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

XX, YY = np.meshgrid(np.arange(X_min, X_max, .02), np.arange(Y_min, Y_max, .02))
Z = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

graph.scatter(X1, X2, c = Y, cmap = graph.cm.rainbow, zorder = 10, edgecolor = 'k', s = 40)
graph.contourf(XX, YY, Z, cmap = graph.cm.rainbow, alpha = 1.0)
graph.contour(XX, YY, Z, colors = 'k', linestyles = '--', alpha=0.5)

graph.title('SVM plot for leaf features')
graph.xlabel('leaf width')
graph.ylabel('leaf length')

graph.show()


# The graph shows three colored zones that the SVM has chosen to group the datapoints in. Color, here, means type of tree. As we can see, the zones correspond reasonably well with the actual tree types of our training data. This means that the SVM can group, for its training data, quite well calculate tree type based on leaf features.
# 
# 
# Now let's do the same using trunk features.
# 
# ### In the cell below replace:
# #### 1. `<addTrunkGirth>` with `'trunk_girth'`
# #### 2. `<addTrunkHeight>` with `'trunk_height'`
# #### Then __run the code__.

# In[ ]:


# Feature 1
###
# REPLACE THE <addTrunkGirth> BELOW WITH 'trunk_girth' (INCLUDING THE QUOTES)
###
trunk_girth = features['trunk_girth'].values
###

# Feature 2
###--- REPLACE THE <addTrunkHeight> BELOW WITH 'trunk_height' (INCLUDING THE QUOTES) ---###
trunk_height = features['trunk_height'].values
###

# Features
trunk_features = np.asarray([trunk_girth, trunk_height]).transpose()

# Fits the SVM model
model = svm.SVC().fit(trunk_features, train_Y)

# Plots the SVM model
X_min, X_max = trunk_features[:, 0].min() - 1, trunk_features[:, 0].max() + 1
Y_min, Y_max = trunk_features[:, 1].min() - 1, trunk_features[:, 1].max() + 1

XX, YY = np.meshgrid(np.arange(X_min, X_max, .02), np.arange(Y_min, Y_max, .02))
Z = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

graph.scatter(trunk_girth, trunk_height, c = train_Y, cmap = graph.cm.rainbow, zorder = 10, edgecolor = 'k', s = 40)
graph.contourf(XX, YY, Z, cmap = graph.cm.rainbow, alpha = 1.0)
graph.contour(XX, YY, Z, colors = 'k', linestyles = '--', alpha = 0.5)

graph.title('SVM plot for leaf features')
graph.xlabel('trunk girth')
graph.ylabel('trunk height')

graph.show()


# Conclusion
# -------
# 
# And that's it! You've made a simple support vector machine that can predict the type of tree based on the leaf and trunk measurements!
# 
# You can go back to the course now and click __'Next Step'__ to move onto how we can test AI models.
