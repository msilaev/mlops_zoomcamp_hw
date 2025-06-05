## Setup Instructions

### 1. Set User ID for Airflow

Before starting Airflow, set the user ID environment variable:

```sh
export AIRFLOW_UID=$(id -u)
```

### 2. Prepare the MLflow Database

To run MLflow, create the database file and set the correct permissions:

```sh
sudo rm -f mlflow.db
sudo touch mlflow.db
sudo chmod 666 mlflow.db
sudo chown $USER:$USER mlflow.db
```

### 3. Launch the Containers

Start all required services using Docker Compose:

```sh
docker compose up -d
```

This will start Airflow, MLflow, Postgres, and Redis containers in the background.