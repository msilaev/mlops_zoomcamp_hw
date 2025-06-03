FROM apache/airflow:2.2.3

ENV AIRFLOW_HOME=/opt/airflow
USER root

# Install system dependencies if needed
RUN apt-get update -qq && apt-get install -y vim

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Create models directory for output artifacts
RUN mkdir -p /opt/airflow/models

USER airflow