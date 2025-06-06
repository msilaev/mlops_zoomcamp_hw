from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor

def batch_train(df, categorical, batch_size=100_000):
    dv = DictVectorizer()
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    first = True

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        dicts = batch[categorical].to_dict(orient='records')
        if first:
            X = dv.fit_transform(dicts)
            first = False
        else:
            X = dv.transform(dicts)
        y = batch['duration'].values
        model.partial_fit(X, y)
    return model, dv

def train_pipeline(**context):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment-1")

    # 1. Load data
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    df = pd.read_parquet(url)
    print(f"Loaded records: {len(df)}")  # Q3

    # 2. Prepare data
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f"Records after preparation: {len(df)}")  # Q4

    # 3. Feature engineering
    #dicts = df[categorical].to_dict(orient='records')

    batch_size = 100000
    dicts_iter = (df.iloc[i:i+batch_size][categorical].to_dict(orient='records')
              for i in range(0, len(df), batch_size))
    dicts = []
    for d in dicts_iter:
        dicts.extend(d)

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    y_train = df['duration'].values

    # 4. Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"Model intercept: {lr.intercept_:.2f}")  # Q5

    # 5. MLflow tracking
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_param("vectorizer", "DictVectorizer")
        mlflow.log_param("features", categorical)
        mlflow.log_metric("intercept", lr.intercept_)
        mlflow.sklearn.log_model(dv, "dict_vectorizer")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="duration_prediction_training",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_pipeline,
    )