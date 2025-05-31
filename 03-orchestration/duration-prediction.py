#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('python -V')


# In[11]:


import pandas as pd


# In[12]:


import pickle


# In[13]:


import xgboost as xgb


# In[14]:


#import seaborn as sns
#import matplotlib.pyplot as plt
#import numpy as np


# In[15]:


from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error


# In[16]:


import mlflow


# In[17]:


mlflow.set_tracking_uri("http://localhost:5000")


# In[18]:


mlflow.set_experiment("nyc-taxi-experiment")


# In[19]:


def read_dataframe(year, month):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02}.parquet'

    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


# In[20]:


#import xgboost as xgb
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#from hyperopt.pyll import scope


# In[21]:


#mlflow.set_tracking_uri("sqlite:///mlflow1.db")
#mlflow.set_experiment("nyc-taxi-experiment_1")



df_val = read_dataframe(year=2021, month=2)
df_train = read_dataframe(year=2021, month=1)


def create_X(df, dv=None):
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:

        dv = DictVectorizer(sparse= True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

X_train, dv = create_X(df_train)
X_val, _ = create_X(df_val, dv=dv)


target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values


def train_model(X_train, y_train, X_val, y_val, dv):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'max_depth': 30,
        'learning_rate': 0.09585,
        'reg_lambda': 0.011074980286498087,
        'reg_alpha': 0.018788520719314586,
        'min_child_weight': 1.06,
        'objective': 'reg:linear',
        'seed': 42
    }

    model = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=30,
        evals=[(valid, 'valid')],
        early_stopping_rounds=50       
    )

    return model


train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)


# In[33]:


from pathlib import Path
model_path = Path("models")
model_path.mkdir(exist_ok=True)


# In[34]:


with mlflow.start_run():

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'max_depth': 30,
        'learning_rate': 0.09585,
        'reg_lambda': 0.011074980286498087,
        'reg_alpha': 0.018788520719314586,
        'min_child_weight': 1.06,
        'objective': 'reg:linear',
        'seed': 42
    }

    mlflow.log_params(best_params)
    model = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=30,
        evals=[(valid, 'valid')],
        early_stopping_rounds=50       
    )

    y_pred = model.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    mlflow.xgboost.log_model(model, artifact_path="models_mlflow")


# In[ ]:




