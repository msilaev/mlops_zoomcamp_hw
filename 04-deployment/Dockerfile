FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Install required Python packages
RUN pip install pandas scikit-learn pyarrow

# Copy your script into the container
COPY homework_week_4.py .

# Set the entrypoint (optional)
# ENTRYPOINT ["python", "predict.py"]