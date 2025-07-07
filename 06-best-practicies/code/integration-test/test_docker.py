import requests 
from deepdiff import DeepDiff

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1654161514.132
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::387546586013:role/lambda-kinesis-role",
            "awsRegion": "eu-west-1",
            "eventSourceARN": "arn:aws:kinesis:eu-west-1:387546586013:stream/ride_events"
        }
    ]
}


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'http://localhost:8081/2015-03-31/functions/function/invocations'
actual_response = (requests.post(url, json=event)).json()
#print(actual_response.json())



expected_response = {'predictions': [{'model': 'ride_duration_prediction_model', 
                                      'version': "m-38915c8da8cf4a0bb96336ab6be26c9f", 
                                      'prediction': {'ride_duration': 18.1689, 'ride_id': 256}}]}


diff = DeepDiff(actual_response, expected_response, ignore_order=True, significant_digits=4)
print(diff)
#assert actual_response == expected_response, f"Expected {expected_response}, but got {actual_response}"
assert "values_changed" not in diff
assert "types_changed" not in diff, f"Type changes found: {diff['type_changes']}"
#docker run -it --rm `
#    -p 8081:8080 `
#    -v "$env:USERPROFILE\.aws:/root/.aws:ro" `
#    -e PREDICTIONS_STREAM_NAME="ride_predictions" `
#    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" `
#    -e TEST_RUN="True" `
#    -e AWS_DEFAULT_REGION="us-east-1" `
#    stream-model-duration:v1