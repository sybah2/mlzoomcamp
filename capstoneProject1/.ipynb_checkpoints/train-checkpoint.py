## importing the needed libaries
print("Importing the needed libraries")
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle

import warnings
warnings.filterwarnings('ignore')

print("Now loading the data for the training of the model")

## Loading the data for the training and modifying the needed columns
data = pd.read_csv("diamonds.csv")


print("Processing and formatting of the data")
data.columns = data.columns.str.lower().str.replace(' ', '_')

string_columns = list(data.dtypes[data.dtypes == 'object'].index)


for col in string_columns:
    data[col] = data[col] = data[col].str.lower().str.replace(" ", "_")

del data['unnamed:_0']

## make a copy of the data for formatting the data
data1 = data.copy()

data1['cut'] = data1['cut'].map({'fair':0, 'good':1, 'very_good':2, 'premium':3, 'ideal':4})
data1['color'] = data1['color'].map({'j':0, 'i':1, 'h':2, 'g':3, 'f':4, 'e':5, 'd':6})
data1['clarity'] = data1['clarity'].map({'i1':0, 'si2':1, 'si1':2, 'vs2':3, 'vs1':4, 'vvs2':5, 'vvs1':6, 'if':7})

## Splitting the data into training, test and validation for the training

print("Splitting the data into training, test and validation sets")
df_full_train, df_test = train_test_split(data1, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)


y_train = df_train.price.values
y_test = df_test.price.values
y_val = df_val.price.values

del df_train['price']
del df_test['price']
del df_val['price']


## Functions below are for training and validation of the model
def train(df, y):
    model = Pipeline([("scalar3",StandardScaler()),
                     ("rf_classifier",RandomForestRegressor())])

    model.fit(df_train, y_train)

    return model


def validation(model, df, y):
    cv_score = cross_val_score(model, df,y,scoring="neg_root_mean_squared_error", cv=10)
    return cv_score.mean() 

print("Training the model")
model = train(df_test, y_train)

print("Checking model performation on training set")
validation(model, df_train,y_train)

print("Checking model performance on the validation set")
validation(model, df_val,y_val)

print("Saving the final model")
output_file = 'final_model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print(f"The model is save to {output_file}")