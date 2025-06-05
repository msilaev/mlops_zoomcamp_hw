from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
from pathlib import Path

TRACKING_SERVER_HOST = "mlflow"
MLFLOW_TRACKING_URI = f"http://{TRACKING_SERVER_HOST}:5000"

def train_pipeline(year, month, **context):


    mlflow.set_tracking_uri("http://mlflow:5000")

    model_local_save_path = Path("/opt/airflow/models")

    model_local_save_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("nyc-taxi-experiment-1") 
    
    def read_dataframe(year, month):
        #url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet'
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet'
               #https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet
        df = pd.read_parquet(url)

        print(df.columns)

        df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        #df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

        #df = df.dropna()
        return df

    def create_X(df, dv=None):
        categorical = ["PU_DO"]
        numerical = ["trip_distance"]

        print(df.columns)
        print(df[categorical + numerical].head())

        dicts = df[categorical + numerical].to_dict(orient='records')
        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
        return X, dv

    def train_model(X_train, y_train, X_val, y_val, dv):
        with mlflow.start_run() as run:
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
                num_boost_round=2,
                evals=[(valid, 'valid')],
                early_stopping_rounds=50
            )
            y_pred = model.predict(valid)
            rmse = (mean_squared_error(y_val, y_pred))**(0.5)
            mlflow.log_metric("rmse", rmse)

            preprocessor_filename = "preprocessor.b"
            local_preprocessor_path = model_local_save_path / preprocessor_filename

            with open(local_preprocessor_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            
                    # *** ADD THIS LINE TO YOUR DAG ***
            print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"MLflow Artifact URI (before logging artifact): {mlflow.get_artifact_uri()}")
                        
            mlflow.log_artifact(str(local_preprocessor_path), artifact_path="preprocessor")
            
#            mlflow.log_artifact(str(model_path / "preprocessor.b"), artifact_path="preprocessor")
            mlflow.xgboost.log_model(model, artifact_path="models_mlflow")   

            return run.info.run_id

    df_train = read_dataframe(year, month)
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1
    df_val = read_dataframe(next_year, next_month)
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"Model trained and logged with run_id: {run_id}")

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
        op_kwargs={'year': 2023, 'month': 3},  # <-- set your default year/month here
    )
    