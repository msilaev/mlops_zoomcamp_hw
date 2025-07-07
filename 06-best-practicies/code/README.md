docker stop $(docker ps -q)
docker rm $(docker ps -aq)

docker build -t stream-model-duration:v2 .

$modelPath = "C:\Users\mikes\Documents\STUDY\mlops-zoomcamp\mlops_zoomcamp_hw\06-best-practicies\code\integration-test\model"

docker run -it --rm `
    -p 8081:8080 `
    -v "$env:USERPROFILE\.aws:/root/.aws:ro" `
    -v "${modelPath}:/app/model" `
    -e PREDICTIONS_STREAM_NAME="ride_predictions" `
    -e MODEL_RUN_ID="Test123" `
    -e MODEL_LOCATION="/app/model" `
    -e TEST_RUN="True" `
    -e AWS_DEFAULT_REGION="us-east-1" `
    stream-model-duration:v2


docker run -it --rm `
    -p 8081:8080 `
    -v "$env:USERPROFILE\.aws:/root/.aws:ro" `
    -e PREDICTIONS_STREAM_NAME="ride_predictions" `
    -e MODEL_RUN_ID="m-38915c8da8cf4a0bb96336ab6be26c9f" `
    -e TEST_RUN="True" `
    -e AWS_DEFAULT_REGION="us-east-1" `
    stream-model-duration:v2


    # Open Git Bash from PowerShell
& "C:\Program Files\Git\bin\bash.exe" run.sh


$LOCAL_TAG = Get-Date -Format "yyyy-MM-dd-HH-mm"
$env:LOCAL_IMAGE_NAME = "stream-model-duration:$LOCAL_TAG"
$env:LOCAL_IMAGE_NAME = "stream-model-duration:2025-07-06-19-31"

docker build -t $LOCAL_IMAGE_NAME .



cd integration-test
docker-compose up -d

sleep 1

python test_docker.py

docker-compose down


kinesis

aws kinesis list-streams

```bash
aws --endpoint-url=http://localhost:4566 \
    kinesis list-streams
```

```bash
aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ride_predictions \
    --shard-count 1
```