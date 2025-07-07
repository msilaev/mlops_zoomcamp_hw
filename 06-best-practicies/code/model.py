import os
import json
import base64

import mlflow
import boto3

def get_model_location(run_id):
    
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location
    
    model_bucket = os.getenv('MODEL_BUCKET', 'mlflow-artifacts-remote-2025')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')

    model_location = f"s3://{model_bucket}/{experiment_id}/models/{run_id}/artifacts/"

    #model_location = f"s3://mlflow-artifacts-remote-2025/1/models/{MODEL_RUN_ID}/artifacts/"
    #s3://mlflow-artifacts-remote-2025/1/models/m-38915c8da8cf4a0bb96336ab6be26c9f/

    return model_location


def load_model(run_id):

#model_uri = f"s3://mlflow-artifacts-remote-2025/1/models/{MODEL_RUN_ID}/artifacts/"
#model = mlflow.pyfunc.load_model(model_uri)

    model_uri = get_model_location(run_id)
    return mlflow.pyfunc.load_model(model_uri)

def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    return json.loads(decoded_data)

def get_kinesis_client():

    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL', None)
    if endpoint_url is None:
        kinesis_client = boto3.client('kinesis')
    else:
        kinesis_client = boto3.client('kinesis', endpoint_url=endpoint_url)         

    return kinesis_client

def create_kinesis_client():
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL', None)
    if endpoint_url is None:
        kinesis_client = boto3.client('kinesis')
        return kinesis_client
        
    return boto3.client('kinesis', endpoint_url=endpoint_url)

def init(prediction_stream_name, run_id,  test_run):
    
    model = load_model(run_id)    
    callbacks = []

    if not test_run:
        kinesis_client = get_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    return ModelService(model, run_id, callbacks)


class KinesisCallback:

    def __init__(self, kinesis_client, prediction_stream_name):
        self.prediction_stream_name = prediction_stream_name
        self.kinesis_client = kinesis_client

    def put_record(self, prediction_event):

        ride_id = prediction_event['prediction']['ride_id']
        self.kinesis_client.put_record(StreamName=self.prediction_stream_name,
                                       Data=json.dumps(prediction_event),   
                                        PartitionKey=str(ride_id),
                                        )


class ModelService:

    def __init__(self, model, model_version=None, callabacks=None):
        self.model = model
        self.model_version = model_version
        self.callbacks = callabacks or []
       
    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self,features):
        return round(float(self.model.predict(features)[0]),4)

    def lambda_handler(self, event):

        predictions = []

        #print("event", event)

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            decoded_data = base64.b64decode(encoded_data).decode('utf-8')
            ride_event = json.loads(decoded_data)
            #print("decoded data", ride_event)

            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = self.prepare_features(ride)

            #print("features", features)
            prediction = self.predict(features) 
            #print("prediction", prediction)

            prediction_event = {
                'model': 'ride_duration_prediction_model',
                'version': self.model_version,
                'prediction': {
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }    

            for callback in self.callbacks:
                callback(prediction_event)  
            
            predictions.append(prediction_event)
        
        return { 'predictions': predictions   } 