#import lambda_function
from pathlib import Path
import model

#def predict(self,features):
#        return float(self.model.predict(features)[0])

class ModelMock:

    def __init__(self, prediction_value):
        self.prediction_value = prediction_value
    
    def predict(self, features):

        return [self.prediction_value]*len(features)

def test_predict():

    model_service = ModelMock(10)
    features = {
        'PU_DO': '130_205',
        'trip_distance': 3.66,
    }
    
    expected_prediction = [10]*len(features)  # Example expected prediction value
    actual_prediction = model_service.predict(features)
    assert actual_prediction == expected_prediction, f"Expected {expected_prediction}, but got {actual_prediction}"


def read_text(file):
    test_directory = Path(__file__).parent

    with open(test_directory / file, 'rt', encoding='utf-8') as f:
        return f.read().strip()

def test_base64_decode():

    base64_input = read_text('data.b64')

    actual_results = model.base64_decode(base64_input)
    expected_results = {
        "ride": {
            "PULocationID": 130, "DOLocationID": 205, "trip_distance": 3.66,
                 },
                 "ride_id": 256,
                 }

    assert actual_results == expected_results

def test_prepare_features():

    model_service = model.ModelService(None)  # Passing None for model since we are testing feature preparation
    ride = {
        'PULocationID': 130,
        'DOLocationID': 205,
        'trip_distance': 3.66,
    }
    
    expected_features = {
        'PU_DO': '130_205',
        'trip_distance': 3.66,
    }
    
    features = model_service.prepare_features(ride)
    
    assert features == expected_features


""" def lambda_handler(self, event):

        predictions = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            decoded_data = base64.b64decode(encoded_data).decode('utf-8')
            ride_event = json.loads(decoded_data)
            #print("decoded data", ride_event)

            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = self.prepare_features(ride)
            prediction = self.predict(features) 

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
        
        return { 'predictions': predictions   }  """


def test_lambda_handler():

    model_mock = ModelMock(10)  # Mocking the model to return a fixed prediction value
    model_version = "1.0"  # Example model version

    model_service = model.ModelService(model_mock, model_version=model_version) 
    event = {
        "Records": [{
                "kinesis": {                    
                    "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                    },              
            }]    
        }

    actual_response = model_service.lambda_handler(event)
    expected_response = {
        'predictions': [
            {
                'model': 'ride_duration_prediction_model',
                'version': model_version,
                'prediction': {
                    'ride_duration': 10.0,  # Assuming the mock model returns 10
                    'ride_id': 256
                }
            }
        ]
    }
    assert actual_response == expected_response, f"Expected {expected_response}, but got {actual_response}"