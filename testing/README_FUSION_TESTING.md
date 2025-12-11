# Fusion Service E2E Testing Guide

This guide explains how to test the fusion service end-to-end using mock model services.

## Overview

The testing setup consists of:
1. **Mock Model Services** - Simulate SER, FER, and Vitals services
2. **E2E Test Script** - Calls fusion service and verifies results
3. **Service Launcher** - Helper to start all mock services

## Prerequisites

1. Fusion service running (`python main.py` on port 8000)
2. Database connection configured (`.env` file with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`)
3. `DEV_USER_ID` set in `.env` file

## Quick Start

### Option 1: Run All Services Together (Recommended for Testing)

```bash
# Terminal 1: Start mock services (all in one process)
python testing/mock_model_services.py

# Terminal 2: Run E2E test
python testing/test_fusion_e2e.py
```

### Option 2: Run Services Separately

```bash
# Terminal 1: Start SER service
python testing/mock_model_services.py ser

# Terminal 2: Start FER service  
python testing/mock_model_services.py fer

# Terminal 3: Start Vitals service
python testing/mock_model_services.py vitals

# Terminal 4: Run E2E test
python testing/test_fusion_e2e.py
```

### Option 3: Use Service Launcher

```bash
# Terminal 1: Start all services using launcher
python testing/run_mock_services.py

# Terminal 2: Run E2E test
python testing/test_fusion_e2e.py
```

## Testing Flow

1. **Start Mock Services**: Run mock services on ports 8005, 8006, 8007
2. **Start Fusion Service**: Ensure `main.py` is running on port 8000
3. **Run Test Script**: Execute `test_fusion_e2e.py`
4. **Enter Signals**: When prompted by mock services, enter signals manually:
   - Format: `emotion:confidence`
   - Examples: `Sad:0.8` or `Sad:0.8,Happy:0.6`
   - Type `auto` for predefined test data
5. **View Results**: Test script displays fusion results and verifies database write

## Input Format

When mock services prompt for input, use these formats:

### Single Signal
```
Sad:0.8
```

### Multiple Signals
```
Sad:0.8,Happy:0.6
```

### With Timestamp
```
Sad:0.8:2025-12-03T10:15:30Z
```

### Auto Mode (Predefined Test Data)
```
auto
```

## Example Test Session

```
[SER] /predict called for user 123e4567-e89b-12d3-a456-426614174000
  Snapshot timestamp: 2025-12-03T10:15:30+08:00
  Window: 60s

  Enter signals for SER (or 'auto' for test data):
  Format: emotion:confidence or emotion:confidence:timestamp
  Multiple: emotion1:conf1,emotion2:conf2
  Examples: Sad:0.8  or  Sad:0.8,Happy:0.6
  Press Enter for empty signals: Sad:0.8,Happy:0.6
  ✓ Parsed 2 signals

[FER] /predict called...
  Enter signals for FER: Sad:0.7
  ✓ Parsed 1 signals

[VITALS] /predict called...
  Enter signals for VITALS: Sad:0.9
  ✓ Parsed 1 signals
```

## Service Ports

- **SER Service**: http://localhost:8005
- **FER Service**: http://localhost:8006
- **Vitals Service**: http://localhost:8007
- **Fusion Service**: http://localhost:8000

## Troubleshooting

### Mock Services Not Starting
- Check if ports 8005, 8006, 8007 are already in use
- Ensure FastAPI and uvicorn are installed

### Fusion Service Not Responding
- Verify `main.py` is running on port 8000
- Check fusion service logs for errors

### No Signals Received
- Ensure you're entering signals when prompted
- Check mock service logs for errors
- Verify network connectivity between services

### Database Write Failed
- Check database connection in `.env`
- Verify `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set
- Check database logs for errors

## Advanced Usage

### Programmatic Signal Setting

You can set signals programmatically using the `/set-signals` endpoint:

```python
import requests

# Set signals for SER service
requests.post("http://localhost:8005/set-signals", json={
    "signals": [
        {
            "user_id": "123e4567-e89b-12d3-a456-426614174000",
            "timestamp": "2025-12-03T10:15:30Z",
            "modality": "speech",
            "emotion_label": "Sad",
            "confidence": 0.8
        }
    ],
    "auto_mode": False
})
```

### Health Checks

Check service health:
```bash
curl http://localhost:8005/health  # SER
curl http://localhost:8006/health  # FER
curl http://localhost:8007/health  # Vitals
curl http://localhost:8000/emotion/health  # Fusion
```

## Files

- `mock_model_services.py` - Mock HTTP servers for model services
- `test_fusion_e2e.py` - E2E test script
- `run_mock_services.py` - Service launcher helper



