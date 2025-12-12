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
from pydantic import BaseModel, ValidationError

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.models import ModelSignal, ModelPredictResponse
from utils import database

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Global state for each service to store signals to return
_service_states = {
    "speech": {"signals": [], "auto_mode": False, "input_string": None},
    "face": {"signals": [], "auto_mode": False, "input_string": None},
    "vitals": {"signals": [], "auto_mode": False, "input_string": None}
}

# Auto-generation state
_auto_generation_enabled = {
    "speech": False,
    "face": False,
    "vitals": False
}
_auto_generation_lock = threading.Lock()
_auto_generation_threads = {}

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


def generate_random_signals(user_id: str, modality: str, snapshot_timestamp: datetime, count: int = None) -> List[ModelSignal]:
    """
    Generate random test signals for a modality.
    
    Args:
        user_id: User ID
        modality: Modality name
        snapshot_timestamp: Base timestamp
        count: Number of signals to generate (default: 1-3 random)
    
    Returns:
        List of ModelSignal objects with random emotions and confidences
    """
    import random
    
    emotions = ["Sad", "Angry", "Happy", "Fear"]
    
    if count is None:
        count = random.randint(1, 3)
    
    signals = []
    malaysia_tz = get_malaysia_timezone()
    
    for i in range(count):
        # Random emotion
        emotion = random.choice(emotions)
        # Random confidence between 0.5 and 0.95
        confidence = round(random.uniform(0.5, 0.95), 2)
        # Random timestamp within last 30 seconds
        offset_seconds = random.randint(0, 30)
        signal_timestamp = snapshot_timestamp - timedelta(seconds=offset_seconds)
        
        signal = ModelSignal(
            user_id=user_id,
            timestamp=signal_timestamp.isoformat(),
            modality=modality,
            emotion_label=emotion,
            confidence=confidence
        )
        signals.append(signal)
    
    return signals


def get_auto_signals(user_id: str, modality: str, snapshot_timestamp: datetime) -> List[ModelSignal]:
    """Generate automatic test signals for a modality (legacy function for backward compatibility)."""
    return generate_random_signals(user_id, modality, snapshot_timestamp, count=2)


def auto_generate_signals_loop(service_name: str, modality: str, interval_seconds: int = 30):
    """
    Background thread that periodically generates new signals and writes to JSON file.
    
    Args:
        service_name: Service name ('ser', 'fer', 'vitals')
        modality: Modality name ('speech', 'face', 'vitals')
        interval_seconds: Generation interval in seconds (default: 30)
    """
    import time
    import random
    
    json_path = get_json_file_path(service_name)
    emotions = ["Sad", "Angry", "Happy", "Fear"]
    
    # Default user ID (can be overridden)
    default_user_id = os.getenv("DEV_USER_ID", "8517c97f-66ef-4955-86ed-531013d33d3e")
    
    logger.info(f"[{service_name.upper()}] Auto-generation started (interval: {interval_seconds}s)")
    
    while _auto_generation_enabled.get(modality, False):
        try:
            # Generate random signals
            now = database.get_current_time_utc8()
            signal_count = random.randint(1, 3)
            signals = generate_random_signals(default_user_id, modality, now, count=signal_count)
            
            # Create ModelPredictResponse
            response = ModelPredictResponse(signals=signals)
            
            # Write to JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(response.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{service_name.upper()}] Generated {len(signals)} signals: {[f'{s.emotion_label}({s.confidence:.2f})' for s in signals]}")
            
            # Wait for next interval
            time.sleep(interval_seconds)
            
        except Exception as e:
            logger.error(f"[{service_name.upper()}] Error in auto-generation: {e}", exc_info=True)
            time.sleep(interval_seconds)


def load_signals_from_json(service_name: str, json_path: str, user_id: str, snapshot_timestamp: datetime, window_seconds: int) -> List[ModelSignal]:
    """
    Load signals from JSON file matching ModelPredictResponse format.
    
    Args:
        service_name: Name of the service (for logging)
        json_path: Path to JSON file
        user_id: User ID to filter signals
        snapshot_timestamp: Snapshot timestamp for time window filtering
        window_seconds: Time window in seconds
    
    Returns:
        List of ModelSignal objects filtered by user_id and time window
    """
    if not os.path.exists(json_path):
        logger.debug(f"[{service_name}] JSON file not found: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate format matches ModelPredictResponse
        try:
            response = ModelPredictResponse(**data)
        except ValidationError as e:
            logger.warning(f"[{service_name}] Invalid JSON format in {json_path}: {e}")
            return []
        
        signals = response.signals
        logger.info(f"[{service_name}] Loaded {len(signals)} signals from {json_path}")
        
        # Filter by user_id
        filtered_signals = [sig for sig in signals if sig.user_id == user_id]
        logger.debug(f"[{service_name}] Filtered to {len(filtered_signals)} signals for user {user_id}")
        
        # Filter by time window
        malaysia_tz = get_malaysia_timezone()
        window_start = snapshot_timestamp.timestamp() - window_seconds
        
        time_filtered_signals = []
        for signal in filtered_signals:
            try:
                signal_timestamp = datetime.fromisoformat(signal.timestamp.replace('Z', '+00:00'))
                if signal_timestamp.tzinfo is None:
                    signal_timestamp = signal_timestamp.replace(tzinfo=malaysia_tz)
                else:
                    signal_timestamp = signal_timestamp.astimezone(malaysia_tz)
                
                signal_ts = signal_timestamp.timestamp()
                if signal_ts >= window_start:
                    time_filtered_signals.append(signal)
            except Exception as e:
                logger.warning(f"[{service_name}] Error parsing signal timestamp: {e}, including signal anyway")
                time_filtered_signals.append(signal)
        
        logger.info(f"[{service_name}] Filtered to {len(time_filtered_signals)} signals within time window")
        return time_filtered_signals
        
    except json.JSONDecodeError as e:
        logger.warning(f"[{service_name}] Invalid JSON in {json_path}: {e}")
        return []
    except Exception as e:
        logger.warning(f"[{service_name}] Error loading JSON from {json_path}: {e}")
        return []


def get_json_file_path(service_name: str) -> str:
    """
    Get the path to the JSON file for a service.
    
    Args:
        service_name: Service name ('ser', 'fer', 'vitals')
    
    Returns:
        Path to JSON file
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to dummy_data directory (same level as this script)
    dummy_data_dir = os.path.join(script_dir, "dummy_data")
    
    # Map service names to JSON file names
    json_files = {
        "ser": "SER_inference.json",
        "fer": "FER_inference.json",
        "vitals": "Vitals_inference.json"
    }
    
    json_filename = json_files.get(service_name.lower(), f"{service_name.upper()}_inference.json")
    return os.path.join(dummy_data_dir, json_filename)


def create_mock_service(service_name: str, port: int, modality: str):
    """Create a FastAPI app for a mock model service."""
    app = FastAPI(title=f"Mock {service_name.upper()} Service")
    
    # Start auto-generation if enabled
    auto_gen_enabled = os.getenv("AUTO_GENERATE_SIGNALS", "true").lower() == "true"
    if auto_gen_enabled:
        with _auto_generation_lock:
            if not _auto_generation_enabled.get(modality, False):
                _auto_generation_enabled[modality] = True
                thread = threading.Thread(
                    target=auto_generate_signals_loop,
                    args=(service_name, modality, 30),  # 30 second interval
                    daemon=True
                )
                thread.start()
                _auto_generation_threads[modality] = thread
                logger.info(f"[{service_name.upper()}] Auto-generation thread started")
    
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
        
        # Check if JSON file mode is enabled (default: true)
        use_json_files = os.getenv("USE_JSON_FILES", "true").lower() == "true"
        
        signals = []
        
        # Priority 1: Try loading from JSON file if enabled
        if use_json_files:
            json_path = get_json_file_path(service_name)
            signals = load_signals_from_json(service_name, json_path, user_id, snapshot_timestamp, window_seconds)
            if signals:
                print(f"\n[{service_name.upper()}] Loaded {len(signals)} signals from JSON file: {json_path}")
        
        # Priority 2: Check if we have signals in state (for programmatic testing)
        if not signals:
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
        
        # Priority 3: Try to get input from file queue or prompt
        if not signals:
            auto_mode = False
            with _state_lock:
                state = _service_states[modality]
                auto_mode = state.get("auto_mode", False)
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
        
        # Filter signals by time window (if not already filtered from JSON)
        # Note: Signals from JSON are already filtered, but signals from other sources need filtering
        if not use_json_files or not signals:
            filtered_signals = []
            window_start = snapshot_timestamp.timestamp() - window_seconds
            malaysia_tz = get_malaysia_timezone()
            
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
            
            signals = filtered_signals
        else:
            # Signals from JSON are already filtered, but we still need to ensure they're ModelSignal objects
            # (they should already be, but double-check)
            pass
        
        # Build response
        response_data = {
            "signals": [sig.dict() for sig in signals]
        }
        
        print(f"[{service_name.upper()}] Returning {len(signals)} signals")
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

