#!/usr/bin/env bash

#set -e  # Exit on any error

cd "$(dirname "$0")" || exit 1

LOCAL_TAG=$(date +"%Y-%m-%d-%H-%M")
export LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"
#export PREDICTIONS_STREAM_NAME="ride_predictions"

echo "Building Docker image: ${LOCAL_IMAGE_NAME}"
# Build from parent directory where Dockerfile is located
docker build -t "${LOCAL_IMAGE_NAME}" ..

echo "Starting services with docker-compose..."
docker-compose up -d
sleep 5

aws --endpoint-url=http://localhost:4566 \
    s3 mb s3://nyc-duration

aws --endpoint-url=http://localhost:4566 s3 ls

# Create the 'in' folder
aws --endpoint-url=http://localhost:4566 s3api put-object --bucket nyc-duration --key in/

# Run the integration test
python integration-test.py

# List files in the bucket
aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/

# Get file size
aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/2023-01.parquet --human-readable

# Set environment variables for batch.py to use localstack
export S3_ENDPOINT_URL="http://localhost:4566"
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

python ../batch.py 2023 1

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down