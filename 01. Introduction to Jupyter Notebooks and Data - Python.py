#!/usr/bin/env python
# coding: utf-8

# # Welcome to Azure Notebooks!
# 
# Python is a free, open source programming language which is extremely popular for statistical analysis and AI.
# 
# Here, we will give you a taste of what using python is like.
# 
# Let's get started. We’ve provided the data for you, and cleaned it up so it’s ready for analysis. You can __move through the steps by clicking on the run button__ just above this notebook.

# Exercise 1 - Introduction To Jupyter Notebooks
# ==========================
# 
# The purpose of this exercise is to get you familiar with using Jupyter Notebooks. Don't worry if you find the coding difficult - this is not a Python course. You will slowly learn more as you go and you definitely don't need to understand every line of code.
# 
# Step 1
# --------
# 
# These notebooks contain places where you can execute code, like below.
# 
# Give it a go. Click on the code below, then press `Run` in the toolbar above (or press __Shift+Enter__) to run the code.

# In[1]:


print("The code ran successfully!")


# If all went well, the code should have printed a message for you.
# 
# At the start of most programming exercises we have to load things to help us do things easily, like creating graphs. 
# 
# Click on the code below, then __hit the `Run` button to load graphing capabilities for later in the exercise__.

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as graph


# Step 2
# --------
# 
# Let's get it to print a message you choose this time. 
# 
# #### Below, write a message between the quotation marks then run the cell.
# 
# It is okay to use spaces, numbers, or letters. Your message should look red. For example, `print("this is my message")`.

# In[3]:


###
# WRITE A MESSAGE BETWEEN THE SPEECH MARKS IN THE LINE BELOW, THEN HIT RUN.
###
print("type something here!")
###

# It's ok to use spaces, numbers, or letters. Your message should look red.
# For example: print("this is my message")


# You will notice hash symbols (`#`). Anything after a `#` is ignored by the computer. This lets us leave notes for you to read so that you understand the code better.

# Step 3
# --------
# 
# Python lets us save things and use them later. In this exercise we will save your message

# In[4]:


###
# WRITE A MESSAGE BETWEEN THE SPEECH MARKS IN THE LINE BELOW, THEN PRESS RUN
###
my_message = "great course"
###

print(my_message) 


# Okay, what's happened here? 
# 
# In the real world we might put something in an envelope (like a letter, or picture). On the envelope we write something (give it a name), like "my_letter_for_alice".
# 
# In a computer, we do something similar. The thing that holds information (like the envelope) is called a **variable**. We also give each one a name. 
# 
# Actually, you've already done this.
# 
# First, you made a message, then you saved it to a **variable** called 'my_message':
# ```
# my_message = "this is my message!"
#               ↑↑↑
#               the message you made
#  
# my_message = "this is my message!"
#           ↑↑↑
#           the equals sign means to save it to the variable on the left
#      
# my_message = "this is my message!"
# ↑↑↑
# this is the name of your variable. They must never have spaces in them.
# ```

# Step 4
# -------
# 
# Let's try using variables again, but save a number inside our variable this time. Remember, the variable is on the *left hand side* of the `=` assignment symbol and is the equivalent of a labelled box. The information on the *right hand side* is the information we want to store inside the variable (or a box in our analogy).
# 
# #### In the cell below replace `<addNumber>` with any number you choose.
# 
# Then __run the code__.

# In[5]:


###
# REPLACE <addNumber> BELOW WITH ANY NUMBER
###
my_first_number = 14
###

print(my_first_number)
print(my_first_number)


# What happened here?
# 
# In the real world, we might then do something with this information. For example, we might choose to read it. We can read it as many times as we like.
# 
# On the computer, we can also do things with this information. Here, you asked the computer to print the message to the screen twice.
# 
# ```
# print(my_first_number) 
# print(my_first_number)
# ```

# How did you do this though?
# 
# ```
# print(....)
# ↑↑↑
# ```
# this is what you are asking the computer to do. It is a **method** called print. There are many methods available. Soon, we will use methods that make graphs.
# ```
# print(....)
#      ↑    ↑
# ```
# methods have round brackets. What you write here between these is given to the method. 
# ```
# print(my_first_number)
#       ↑↑↑
# ```
# In this case, we gave it 'my_first_number', and it took it and printed it to the screen.               
#       
# 
# Step 5
# -------
# 
# Ok, let's make a graph from some data.
# 
# #### In the cell below replace the `<addNumber>`'s with any number you choose
# 
# Then __run the code__ to make a graph.

# In[6]:


# These are our x values
x_values = [1, 2, 3]

###
# BELOW INSIDE THE SQUARE BRACKETS, REPLACE THE <addNumber>'S WITH EACH WITH A NUMBER
###
y_values = [3, 1, 7]
###

# When you've done that, run the cell
# For example, you could change like this: y_values = [3, 1, 7]

# This makes a bar graph. We give it our x and y values
graph.bar(x_values, y_values)


# This is very simple, but here x and y are our data.
# 
# If you'd like, have a play with the code:
# * change x and y values and see how the graph changes. Make sure they have the same count of numbers in them.
# * change `graph.bar` to `graph.scatter` to change the type of graph
# 
# 
# Step 6
# ----------------
# 
# From time to time, we will load data from text files, rather than write it into the code. You can't see these text files in your browser because they are saved on the server running this website. We can load them using code, though. Let's load one up, look at it, then graph it.
# 
# #### In the cell below write `print(data.head())` then __run the code__.

# In[7]:


import pandas as pd

# The next line loads information about chocolate bars and saves it in a variable called 'data'
data = pd.read_csv('Data/chocolate data.txt', index_col = False, sep = '\t')

### 
# WRITE print(data.head()) BELOW TO PREVIEW THE DATA ---###
###
print(data.head())
###


# Each row (horizontal) shows information about one chocolate bar. For example, the first chocolate bar was:
# * 185 grams
# * 65% cocoa
# * 11% sugar
# * 24% milk
# * and a customer said they were 47% happy with it
# 
# We would probably say that our chocolate bar features were weight, cocoa %, sugar % and milk %
# 
# Conclusion
# ----------------
# 
# __Well done__ that's the end of programming exercise one.
# 
# You can now go back to the course and click __'Next Step'__ to move onto some key concepts of AI - models and error.
# 
# 
# Optional Step 7
# ----------------
# When we say "optional" we mean exercises that might help you learn, but you don't have to do. 
# 
# We can graph some of these features in scatter plot. Let's put cocoa_percent on the x-axis and customer happiness on the y axis.
# 
# #### In the cell below replace `<addYValues>` with `customer_happiness` and then __run the code__.

# In[8]:


x_values = data.cocoa_percent

###
# REPLACE <addYValues> BELOW WITH customer_happiness
###
y_values = data.customer_happiness
###

graph.scatter(x_values, y_values)


# In this graph, every chocolate bar is one point. Later, we will analyse this data with AI.
