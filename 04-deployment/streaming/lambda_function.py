import json
import base64
import boto3
import os

import mlflow
from mlflow.tracking import MlflowClient

from flask import Flask, request, jsonify


RUN_ID = 'fd2154d35755423d9bb2701da304a056'
MODEL_RUN_ID = 'm-38915c8da8cf4a0bb96336ab6be26c9f'

AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')


MODEL_RUN_ID = os.getenv('MODEL_RUN_ID', 'm-38915c8da8cf4a0bb96336ab6be26c9f') 
#logged_model =f'runs:/{RUN_ID}/model'
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#loaded_model = mlflow.pyfunc.load_model(logged_model)

#model_uri = "s3://mlflow-artifacts-remote-2025/1/{RUN_ID}/artifacts/model"
model_uri = f"s3://mlflow-artifacts-remote-2025/1/models/{MODEL_RUN_ID}/artifacts/"
model = mlflow.pyfunc.load_model(model_uri)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']

    return features

def predict(features):
    #X = dv.transform(features)
    #y_pred = model.predict(X)
    #y_pred = loaded_model.predict(features)

    #print(f'Predicted duration: {y_pred}')

    return float(model.predict(features)[0])

#kinesis_client = boto3.client('kinesis')
kinesis_client = boto3.client('kinesis', region_name=AWS_REGION)


PREDICTIONS_STREAM_NAME =  os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
#def prepare_features(ride):
#    features = {}
    #features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    #features['trip_distance'] = ride['trip_distance']

#    return features



def lambda_handler(event, context):
    # TODO implement
    
    #print(json.dumps(event))
    print(event)

    predictions = []

    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
        print("decoded data", ride_event)

        ride = ride_event['ride']
        ride_id = ride_event['ride_id']

        features = prepare_features(ride)
        prediction = predict(features) 

        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': '123',
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id
            }
        }      
        
        predictions.append({
            'ride_duration': prediction,
            'ride_id': ride_id})
        #prediction = 10
        #ride_id = 1

        #kinesis_client.put_record(
        #    StreamName=PREDICTIONS_STREAM_NAME,
        #    Data=json.dumps(prediction_event),
        #    PartitionKey=str(ride_id)
        #)

        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )

    return {
        'ride_duration': prediction,
        'ride_id': ride_id
    }
    