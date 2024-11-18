### Loading the packages needed
print("Loading the packges")
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import roc_curve


import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
#!pip install xgboost
import xgboost as xgb

output_file = 'xboos_model.bin'

## Reading the data
df = pd.read_csv("data/bank_churn_data.csv")
del df['customer_id']


## data preparation
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = df_train.churn.values
y_test = df_test.churn.values
y_val = df_val.churn.values
y_full_train = df_full_train.churn.values

del df_train['churn']
del df_test['churn']
del df_val['churn']
del df_full_train['churn']


### Model parameters
xgb_params = {
    'eta': 0.01, 
    'max_depth': 8,
    'min_child_weight': 30,
     
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
 
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
## Funcation to train the model
def train(df, y, params):
    df = df_full_train.to_dict(orient='records')
 
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(df)

    feature_names = list(dv.get_feature_names_out())

    feature_names = list(dv.get_feature_names_out())
    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=feature_names)
 
    model = xgb.train(params, dfulltrain, num_boost_round=200,
                  verbose_eval=5,)

    return (dv, model)

## Function to validate the model
def valid(model, dv, df, y):
    dicts_test = df.to_dict(orient='records')
    X_test = dv.transform(dicts_test)

    feature_names = list(dv.get_feature_names_out())
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    y_pred = model.predict(dtest)
    return roc_auc_score(y, y_pred)

print(f"Training the model with this parameters {xgb_params}")
## Train the model
dv, model = train(df_full_train, y_train, xgb_params)

print("Running the model on the test dataset")
## Running the model on the test dataset
auc = valid(model=model, dv=dv, df=df_test, y=y_test)
print(f"The model AUC on the test dataset is: {auc} ")

print("Saving the model")
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"The model is save to {output_file}")