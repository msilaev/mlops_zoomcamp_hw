FROM apache/airflow:2.2.3

ENV AIRFLOW_HOME=/opt/airflow
USER root

# Install system dependencies if needed
#RUN apt-get update -qq && apt-get install -y vim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cmake \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Create models directory for output artifacts
RUN mkdir -p /opt/airflow/models
RUN mkdir -p /opt/airflow/mlruns && chown -R airflow: /opt/airflow/mlruns


USER airflow