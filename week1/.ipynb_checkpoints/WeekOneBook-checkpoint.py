#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#! wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv


# ## Q1 Pandas version

# In[3]:


pd.__version__


# ## Read in the data for the following questions

# In[4]:


df = pd.read_csv('laptops.csv')


# In[19]:


df.head()


# In[28]:


## Replace the spaces in the header with under score and make it to lower case
df.columns = df.columns.str.lower().str.replace(" ", "_")


# In[29]:


df.head()


# In[30]:


sns.histplot(df.final_price)


# In[32]:


## Examine the data types of the data frame
df.dtypes


# ## Q2 Records count

# In[6]:


## Number of rows. Determine the shape of the matrix and get the number of row [0]
df.shape[0]


# ## Q3 Laptop brands

# In[33]:


### Count the number of laptop brands
len(list(df.brand.unique()))


# ## Q4 Missing values

# In[34]:


## Select the number of columns with NA in the dataframe, turn into a list and determine the lenght of the list

len(list(df.isna().sum()[df.isna().sum() > 1]))


# ## Q5 Maximum final price

# In[35]:


max(df[df.brand == "Dell"].final_price)


# ## Q6 Median value of Screen

# ### Find the median value of Screen column in the dataset.

# In[37]:


median_value_with_NA = df.screen.median()
print(median_value_with_NA)


# ### Next, calculate the most frequent value of the same Screen column.

# In[38]:


most_common_size = df.screen.mode()
print(most_common_size[0])


# ### Use fillna method to fill the missing values in Screen column with the most frequent value from the previous step.

# In[39]:


df.screen.fillna(most_common_size[0],inplace=True)


# ### Now, calculate the median value of Screen once again

# In[40]:


df.isna().sum()


# In[41]:


median_value_without_NA = df.screen.median()
print(median_value_without_NA)


# ## Q7 Sum of weights
# 
# 1. Select all the "Innjoo" laptops from the dataset.
# 2. Select only columns RAM, Storage, Screen.
# 3. Get the underlying NumPy array. Let's call it X.
# 4. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# 5. Compute the inverse of XTX.
# 6. Create an array y with values [1100, 1300, 800, 900, 1000, 1100].
# 7. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# 8. What's the sum of all the elements of the result?

# 

# In[43]:


## 1 and 2
Innjoo = df[df.brand == 'Innjoo'][["ram","storage","screen"]]

## 3
X = Innjoo.to_numpy()
## 4 and 5
XT = X.T
XTX = np.dot(XT, X)
## 6
XTX_inv = np.linalg.inv(XTX)


# In[44]:


y = [1100, 1300, 800, 900, 1000, 1100]


# In[45]:


w = np.dot(np.dot(XTX_inv,XT),y)


# In[46]:


w


# In[47]:


w.sum()


# In[ ]:




