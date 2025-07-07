
import os
import model

#RUN_ID = 'fd2154d35755423d9bb2701da304a056'
MODEL_RUN_ID =  os.getenv("MODEL_RUN_ID") #'m-38915c8da8cf4a0bb96336ab6be26c9f'
#AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
#MODEL_RUN_ID = os.getenv('MODEL_RUN_ID', 'm-38915c8da8cf4a0bb96336ab6be26c9f') 
#logged_model =f'runs:/{RUN_ID}/model'
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

#MLFLOW_TRACKING_URI = 'http://localhost:5000'
PREDICTIONS_STREAM_NAME =  os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')


model_servide = model.init(prediction_stream_name = PREDICTIONS_STREAM_NAME,
                           run_id=MODEL_RUN_ID,
                          test_run=TEST_RUN)

def lambda_handler(event, context):
   
   return model_servide.lambda_handler(event)