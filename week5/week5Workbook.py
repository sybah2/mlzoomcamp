#!/usr/bin/env python
# coding: utf-8

# In[4]:


## Import the modules needed for the analys
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# In[5]:


df = pd.read_csv('customer_churn.csv')

## make all column names the same format
df.columns = df.columns.str.lower().str.replace(' ', '_')
## Make all values the same format (only strings values)
string_var = list(df.dtypes[df.dtypes == 'object'].index)
for var in string_var:
    df[var] = df[var].str.lower().str.replace(' ', '_')
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[6]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[7]:


numeric = ['tenure', 'monthlycharges','totalcharges' ]

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[8]:


def train(df, y, C=10):
    dicts = df[categorical + numeric].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y)

    return dv, model


# In[9]:


def predict(df, dv, model):
    dicts = df[categorical + numeric].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred
    


# In[10]:


C=1.0
n_splits = 5


# In[11]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):     
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)


    roc = roc_auc_score(y_val, y_pred)
    scores.append(roc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores),np.std(scores)))


# In[12]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
roc = roc_auc_score(y_test, y_pred)
roc


# In[13]:


import pickle


# In[14]:


output_file = f'model_C={C}.bin'


# In[15]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# In[1]:


import pickle


# In[2]:


model_file = 'model_C=1.0.bin'


# In[3]:


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


dv, model


# In[6]:


customer = {'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'yes',
  'dependents': 'no',
  'phoneservice': 'no',
  'multiplelines': 'no_phone_service',
  'internetservice': 'dsl',
  'onlinesecurity': 'no',
  'onlinebackup': 'yes',
  'deviceprotection': 'no',
  'techsupport': 'no',
  'streamingtv': 'no',
  'streamingmovies': 'no',
  'contract': 'month-to-month',
  'paperlessbilling': 'yes',
  'paymentmethod': 'electronic_check',
  'tenure': 1,
  'monthlycharges': 29.85,
  'totalcharges': 29.85}


# In[8]:


X  = dv.transform([customer])


# In[9]:


model.predict_proba(X)[0,1]


# In[ ]:




