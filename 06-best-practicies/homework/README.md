# MLOps Zoomcamp Homework - Best Practices

## Configuring Unit Tests (PowerShell)

### 1. Install pytest as development dependency
```powershell
pipenv install --dev pytest
```

### 2. Create pytest.ini configuration file
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v
norecursedirs = ../*
```

### 3. Run specific unit test
```powershell
python -m pytest tests/test_batch.py::test_prepare_data -v
```

---

## Integration Test

### 1. Set environment variables for S3 configuration
```powershell
$env:INPUT_FILE_PATTERN = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
$env:OUTPUT_FILE_PATTERN = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
$env:S3_ENDPOINT_URL = "http://localhost:4566"
```

### 2. Run batch processing with localstack
```powershell
python batch.py 2023 3
```

### 3. Running integration test script
```powershell
cd integration-test
& "C:\Program Files\Git\bin\bash.exe" run.sh
```

---

## Prerequisites

- Python 3.11+
- pipenv
- Docker Desktop
- AWS CLI
- Git Bash (for Windows)

---

## Project Structure

```
homework/
├── batch.py                 # Main batch processing script
├── model.bin               # Trained model file
├── pytest.ini             # pytest configuration
├── Pipfile                 # Python dependencies
├── tests/
│   └── test_batch.py       # Unit tests
└── integration-test/
    ├── docker-compose.yaml # Localstack configuration
    ├── integration-test.py # Integration test script
    └── run.sh              # Test runner script
```

---
