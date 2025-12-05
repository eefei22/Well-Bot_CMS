#!/usr/bin/env python3
"""
Mock Model Services for Fusion Service Testing

This script runs three FastAPI servers (SER, FER, Vitals) on ports 8005, 8006, 8007
that simulate emotion recognition model services. Each service accepts manual input
via terminal prompts when /predict is called.

Usage:
    python testing/mock_model_services.py [service_name]
    
    service_name: 'ser', 'fer', 'vitals', or 'all' (default: 'all')
"""

import sys
import os
import asyncio
import threading
import tempfile
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.models import ModelSignal, ModelPredictResponse
from utils import database

# Global state for each service to store signals to return
_service_states = {
    "speech": {"signals": [], "auto_mode": False, "input_string": None},
    "face": {"signals": [], "auto_mode": False, "input_string": None},
    "vitals": {"signals": [], "auto_mode": False, "input_string": None}
}

# Lock for thread-safe access
_state_lock = threading.Lock()

# File-based input queue (for when stdin isn't available)
_input_queue_dir = tempfile.gettempdir()
_input_queue_files = {
    "speech": os.path.join(_input_queue_dir, "mock_ser_input.json"),
    "face": os.path.join(_input_queue_dir, "mock_fer_input.json"),
    "vitals": os.path.join(_input_queue_dir, "mock_vitals_input.json")
}


def get_malaysia_timezone():
    """Get Malaysia timezone (UTC+8)."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Kuala_Lumpur")
    except (ImportError, Exception):
        try:
            import pytz
            return pytz.timezone("Asia/Kuala_Lumpur")
        except ImportError:
            from datetime import timezone, timedelta
            return timezone(timedelta(hours=8))


def parse_signal_input(input_str: str, user_id: str, modality: str, snapshot_timestamp: datetime) -> List[ModelSignal]:
    """
    Parse user input string into ModelSignal objects.
    
    Format: "emotion:confidence" or "emotion:confidence:timestamp"
    Multiple signals: "emotion1:conf1,emotion2:conf2"
    
    Args:
        input_str: Input string from user
        user_id: User ID from request
        modality: Modality name ("speech", "face", "vitals")
        snapshot_timestamp: Snapshot timestamp for default timestamps
    
    Returns:
        List of ModelSignal objects
    """
    signals = []
    
    if not input_str or input_str.strip() == "":
        return signals
    
    # Split by comma for multiple signals
    signal_strings = [s.strip() for s in input_str.split(",")]
    
    malaysia_tz = get_malaysia_timezone()
    
    for signal_str in signal_strings:
        parts = signal_str.split(":")
        
        if len(parts) < 2:
            print(f"  ⚠ Invalid format: {signal_str} (expected emotion:confidence)")
            continue
        
        emotion_label_raw = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except ValueError:
            print(f"  ⚠ Invalid confidence: {parts[1]}")
            continue
        
        # Normalize emotion label (case-insensitive)
        emotion_label_lower = emotion_label_raw.lower()
        emotion_mapping = {
            "angry": "Angry",
            "sad": "Sad",
            "happy": "Happy",
            "fear": "Fear"
        }
        emotion_label = emotion_mapping.get(emotion_label_lower)
        
        # Validate emotion
        if emotion_label is None:
            valid_emotions = ["Angry", "Sad", "Happy", "Fear"]
            print(f"  ⚠ Invalid emotion: {emotion_label_raw} (must be one of {valid_emotions})")
            print(f"     Note: Emotion labels are case-insensitive (sad=Sad, happy=Happy, etc.)")
            continue
        
        # Parse timestamp if provided, otherwise use snapshot_timestamp
        if len(parts) >= 3:
            try:
                timestamp_str = parts[2].strip()
                signal_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if signal_timestamp.tzinfo is None:
                    signal_timestamp = signal_timestamp.replace(tzinfo=malaysia_tz)
                else:
                    signal_timestamp = signal_timestamp.astimezone(malaysia_tz)
            except Exception as e:
                print(f"  ⚠ Invalid timestamp: {parts[2]}, using snapshot timestamp")
                signal_timestamp = snapshot_timestamp
        else:
            signal_timestamp = snapshot_timestamp
        
        signal = ModelSignal(
            user_id=user_id,
            timestamp=signal_timestamp.isoformat(),
            modality=modality,
            emotion_label=emotion_label,
            confidence=confidence
        )
        signals.append(signal)
    
    return signals


def get_auto_signals(user_id: str, modality: str, snapshot_timestamp: datetime) -> List[ModelSignal]:
    """Generate automatic test signals for a modality."""
    malaysia_tz = get_malaysia_timezone()
    
    auto_data = {
        "ser": [
            {"emotion_label": "Sad", "confidence": 0.82},
            {"emotion_label": "Happy", "confidence": 0.60}
        ],
        "fer": [
            {"emotion_label": "Sad", "confidence": 0.75}
        ],
        "vitals": [
            {"emotion_label": "Sad", "confidence": 0.88}
        ]
    }
    
    signals = []
    for data in auto_data.get(modality, []):
        signal = ModelSignal(
            user_id=user_id,
            timestamp=snapshot_timestamp.isoformat(),
            modality=modality,
            emotion_label=data["emotion_label"],
            confidence=data["confidence"]
        )
        signals.append(signal)
    
    return signals


def create_mock_service(service_name: str, port: int, modality: str):
    """Create a FastAPI app for a mock model service."""
    app = FastAPI(title=f"Mock {service_name.upper()} Service")
    
    @app.post("/predict")
    async def predict(request: Request):
        """Predict endpoint that accepts manual input."""
        global _service_states
        
        # Parse request body
        body = await request.json()
        user_id = body.get("user_id", "")
        snapshot_timestamp_str = body.get("snapshot_timestamp", "")
        window_seconds = body.get("window_seconds", 60)
        
        # Parse snapshot timestamp
        try:
            snapshot_timestamp = datetime.fromisoformat(snapshot_timestamp_str.replace('Z', '+00:00'))
            malaysia_tz = get_malaysia_timezone()
            if snapshot_timestamp.tzinfo is None:
                snapshot_timestamp = snapshot_timestamp.replace(tzinfo=malaysia_tz)
            else:
                snapshot_timestamp = snapshot_timestamp.astimezone(malaysia_tz)
        except Exception:
            snapshot_timestamp = database.get_current_time_utc8()
        
        # Check if we have signals in state
        with _state_lock:
            state = _service_states[modality]
            signals = state["signals"].copy()
            auto_mode = state["auto_mode"]
            input_string = state.get("input_string")
            # Clear state after use
            state["signals"] = []
            state["auto_mode"] = False
            state["input_string"] = None
        
        # If we have input_string in state, parse it first
        if input_string:
            signals = parse_signal_input(input_string, user_id, modality, snapshot_timestamp)
            print(f"\n[{service_name.upper()}] Using input from state: {len(signals)} signals")
        
        # If no signals in state, try to get input from file queue or prompt
        if not signals:
            if auto_mode:
                signals = get_auto_signals(user_id, modality, snapshot_timestamp)
                print(f"\n[{service_name.upper()}] Auto mode: Returning {len(signals)} signals")
            else:
                # Try to read from input queue file first
                input_file = _input_queue_files.get(modality)
                user_input = None
                
                if input_file and os.path.exists(input_file):
                    try:
                        with open(input_file, 'r') as f:
                            queue_data = json.load(f)
                            user_input = queue_data.get("input", "").strip()
                            # Delete file after reading
                            os.remove(input_file)
                            print(f"\n[{service_name.upper()}] Read input from queue file")
                    except Exception as e:
                        print(f"\n[{service_name.upper()}] Error reading queue file: {e}")
                
                # If no input from file, try stdin (works if running in foreground)
                if not user_input:
                    print(f"\n[{service_name.upper()}] /predict called for user {user_id}")
                    print(f"  Snapshot timestamp: {snapshot_timestamp.isoformat()}")
                    print(f"  Window: {window_seconds}s")
                    print(f"\n  Enter signals for {service_name.upper()} (or 'auto' for test data):")
                    print(f"  Format: emotion:confidence or emotion:confidence:timestamp")
                    print(f"  Multiple: emotion1:conf1,emotion2:conf2")
                    print(f"  Examples: sad:0.8  or  sad:0.8,happy:0.6")
                    print(f"  Press Enter for empty signals: ", end="", flush=True)
                    
                    try:
                        user_input = input().strip()
                    except (EOFError, KeyboardInterrupt, OSError):
                        # stdin not available (running in background thread)
                        print(f"\n  ⚠ Cannot read from stdin (service running in background)")
                        print(f"  💡 Tip: Use /set-signals endpoint or write to {input_file}")
                        user_input = None
                
                # Process input
                if user_input:
                    if user_input.lower() == "auto":
                        signals = get_auto_signals(user_id, modality, snapshot_timestamp)
                        print(f"  ✓ Auto mode: Returning {len(signals)} signals")
                    else:
                        signals = parse_signal_input(user_input, user_id, modality, snapshot_timestamp)
                        print(f"  ✓ Parsed {len(signals)} signals")
                else:
                    print(f"  ✓ Returning empty signals (no input provided)")
        
        # Filter signals by time window
        filtered_signals = []
        window_start = snapshot_timestamp.timestamp() - window_seconds
        
        for signal in signals:
            try:
                signal_timestamp = datetime.fromisoformat(signal.timestamp.replace('Z', '+00:00'))
                if signal_timestamp.tzinfo is None:
                    signal_timestamp = signal_timestamp.replace(tzinfo=malaysia_tz)
                else:
                    signal_timestamp = signal_timestamp.astimezone(malaysia_tz)
                
                signal_ts = signal_timestamp.timestamp()
                if signal_ts >= window_start:
                    filtered_signals.append(signal)
            except Exception:
                # Include signal if timestamp parsing fails
                filtered_signals.append(signal)
        
        # Build response
        response_data = {
            "signals": [sig.dict() for sig in filtered_signals]
        }
        
        print(f"[{service_name.upper()}] Returning {len(filtered_signals)} signals")
        return JSONResponse(content=response_data)
    
    @app.post("/set-signals")
    async def set_signals(request: Request):
        """Set signals to return on next /predict call (for programmatic testing)."""
        global _service_states
        
        body = await request.json()
        signals_data = body.get("signals", [])
        auto_mode = body.get("auto_mode", False)
        input_string = body.get("input", None)  # Allow setting input string directly
        
        with _state_lock:
            if input_string:
                # Store input string to be parsed later
                _service_states[modality]["input_string"] = input_string
                _service_states[modality]["signals"] = []
            else:
                _service_states[modality]["signals"] = signals_data
                _service_states[modality]["input_string"] = None
            _service_states[modality]["auto_mode"] = auto_mode
        
        return {"status": "ok", "message": f"Set signals for {service_name}"}
    
    @app.post("/set-input")
    async def set_input(request: Request):
        """Set input string to be parsed on next /predict call."""
        global _input_queue_files
        
        body = await request.json()
        input_string = body.get("input", "")
        
        # Write to queue file
        input_file = _input_queue_files.get(modality)
        if input_file:
            try:
                with open(input_file, 'w') as f:
                    json.dump({"input": input_string}, f)
                return {"status": "ok", "message": f"Input queued for {service_name}", "file": input_file}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        return {"status": "error", "message": "No input file configured"}
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": service_name}
    
    return app


def run_service(service_name: str, port: int, modality: str):
    """Run a mock service on the specified port."""
    app = create_mock_service(service_name, port, modality)
    print(f"Starting {service_name.upper()} mock service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main():
    """Main entry point."""
    import sys
    
    service_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    services = {
        "ser": (8005, "speech"),
        "fer": (8006, "face"),
        "vitals": (8007, "vitals")
    }
    
    if service_arg.lower() == "all":
        print("=" * 80)
        print("MOCK MODEL SERVICES")
        print("=" * 80)
        print("Starting all services:")
        print("  SER:    http://localhost:8005")
        print("  FER:    http://localhost:8006")
        print("  Vitals: http://localhost:8007")
        print("=" * 80)
        print("\nNote: This will start all services in the same process.")
        print("For production testing, run each service separately.\n")
        
        # Run all services in separate threads
        threads = []
        for name, (port, modality) in services.items():
            thread = threading.Thread(
                target=run_service,
                args=(name, port, modality),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Keep main thread alive
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down services...")
            sys.exit(0)
    else:
        if service_arg.lower() not in services:
            print(f"Unknown service: {service_arg}")
            print(f"Available: {', '.join(services.keys())}, all")
            sys.exit(1)
        
        port, modality = services[service_arg.lower()]
        run_service(service_arg.lower(), port, modality)


if __name__ == "__main__":
    main()

