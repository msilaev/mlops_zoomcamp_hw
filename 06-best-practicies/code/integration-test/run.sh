#!/usr/bin/env bash

#set -e  # Exit on any error

cd "$(dirname "$0")" || exit 1

LOCAL_TAG=$(date +"%Y-%m-%d-%H-%M")
export LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"
export PREDICTIONS_STREAM_NAME="ride_predictions"

echo "Building Docker image: ${LOCAL_IMAGE_NAME}"
# Build from parent directory where Dockerfile is located
docker build -t "${LOCAL_IMAGE_NAME}" ..

echo "Starting services with docker-compose..."
docker-compose up -d
sleep 5
aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --shard-count 1 || echo "Stream already exists, continuing..."

# Wait for container to be ready
echo "Waiting for container to start..."
sleep 1

# Run the test
echo "Running integration test..."
pipenv run python test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi


pipenv run python test_kinesis.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi


docker-compose down