import argparse
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np

def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    year = args.year
    month = args.month

    val_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df_val = read_dataframe(val_path)

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    # Load DictVectorizer and model from files already in the image
    with open('model2.bin', 'rb') as f_in:
        dv = pickle.load(f_in)
    with open('model.bin', 'rb') as f_in:
        model = pickle.load(f_in)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_pred = model.predict(X_val)

    # Create ride_id for validation set
    df_val['ride_id'] = f'{year:04d}/{month:02d}_' + df_val.index.astype('str')

    df_result = pd.DataFrame({
        'ride_id': df_val['ride_id'],
        'prediction': y_pred
    })

    print(f"Mean predicted duration: {df_result['prediction'].mean()}")