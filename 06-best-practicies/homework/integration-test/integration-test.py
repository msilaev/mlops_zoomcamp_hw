import os
import pandas as pd
from datetime import datetime 

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def save_data():
    # Create test data (same as Q3)
    data = [(None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    # Set up S3 options for localstack
    options = {
        'client_kwargs': {
            'endpoint_url': 'http://localhost:4566'
        }
    }

    # Save to S3 as January 2023 data
    input_file = 's3://nyc-duration/in/2023-01.parquet'
    
    print(f"Saving test data to: {input_file}")
    
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )
    
    print("✅ Test data saved successfully!")

if __name__ == '__main__':
    save_data()