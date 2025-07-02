import requests 

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
response = requests.post(url, json=event)
print(response.json())

#docker run -it --rm `
#    -p 8081:8080 `
#    -v "$env:USERPROFILE\.aws:/root/.aws:ro" `
#    -e PREDICTIONS_STREAM_NAME="ride_predictions" `
#    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" `
#    -e TEST_RUN="True" `
#    -e AWS_DEFAULT_REGION="us-east-1" `
#    stream-model-duration:v1