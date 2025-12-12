"""
Orchestrator Layer for Fusion Service

This module orchestrates the complete emotion fusion flow:
1. Parse and validate request
2. Call model clients in parallel
3. Filter signals by time window
4. Check minimum signals requirement
5. Run fusion logic
6. Write to database
7. Return response
"""

import logging
import asyncio
from typing import Optional, Dict, List, Union
from datetime import datetime

from fusion.models import EmotionSnapshotRequest, FusedEmotionResponse, NoSignalsResponse, SignalUsed, ModelSignal, EmotionSnapshotDemoRequest
from fusion.model_clients import SERClient, FERClient, VitalsClient
from fusion.fusion_logic import fuse_signals
from fusion.config_loader import load_config
from utils import database
from utils import activity_logger

logger = logging.getLogger(__name__)

# Load configuration
_config = load_config()
_time_window = _config.get("time_window_seconds", 60)


def validate_snapshot_request(request: EmotionSnapshotRequest) -> None:
    """
    Validate emotion snapshot request.
    
    Args:
        request: EmotionSnapshotRequest to validate
    
    Raises:
        ValueError: If validation fails
    """
    import uuid
    
    # Validate user_id is a valid UUID
    try:
        uuid.UUID(request.user_id)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid user_id format: '{request.user_id}'. Must be a valid UUID.")


async def process_emotion_snapshot(request: EmotionSnapshotRequest) -> Union[FusedEmotionResponse, NoSignalsResponse]:
    """
    Process an emotion snapshot request.
    
    This orchestrates the complete flow:
    1. Parse and validate request
    2. Call model clients in parallel (asyncio)
    3. Filter signals by time window (already done by model clients)
    4. Check minimum signals requirement
    5. Run fusion logic
    6. Write to database
    7. Return response
    
    Args:
        request: EmotionSnapshotRequest with user_id and optional parameters
    
    Returns:
        FusedEmotionResponse with fused result, or NoSignalsResponse if no valid signals
    """
    logger.info(f"Processing emotion snapshot request for user {request.user_id}")
    
    try:
        # Step 1: Validate request
        validate_snapshot_request(request)
        
        # Determine snapshot timestamp
        if request.timestamp:
            # Parse provided timestamp
            snapshot_timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
            # Ensure UTC+8 timezone
            malaysia_tz = database.get_malaysia_timezone()
            if snapshot_timestamp.tzinfo is None:
                snapshot_timestamp = snapshot_timestamp.replace(tzinfo=malaysia_tz)
            else:
                snapshot_timestamp = snapshot_timestamp.astimezone(malaysia_tz)
        else:
            # Use current time in UTC+8
            snapshot_timestamp = database.get_current_time_utc8()
        
        # Get window_seconds from request options or config
        window_seconds = request.options.window_seconds if request.options else None
        window_seconds = window_seconds or _time_window
        
        # Get timeout override if provided
        timeout_override = request.options.timeout_seconds if request.options else None
        
        logger.info(f"Snapshot timestamp: {snapshot_timestamp.isoformat()}, window: {window_seconds}s")
        
        # Track start time for duration
        fusion_start_time = datetime.now()
        
        # Step 2: Call model services in parallel
        # Initialize clients with optional timeout override
        ser_client = SERClient(timeout=timeout_override)
        fer_client = FERClient(timeout=timeout_override)
        vitals_client = VitalsClient(timeout=timeout_override)
        
        logger.info(f"Calling model services in parallel for user {request.user_id}...")
        logger.info(f"  SER endpoint: {ser_client.service_url}/predict")
        logger.info(f"  FER endpoint: {fer_client.service_url}/predict")
        logger.info(f"  Vitals endpoint: {vitals_client.service_url}/predict")
        
        # Call all clients concurrently
        ser_task = ser_client.predict(request.user_id, snapshot_timestamp, window_seconds)
        fer_task = fer_client.predict(request.user_id, snapshot_timestamp, window_seconds)
        vitals_task = vitals_client.predict(request.user_id, snapshot_timestamp, window_seconds)
        
        ser_signals, fer_signals, vitals_signals = await asyncio.gather(
            ser_task,
            fer_task,
            vitals_task,
            return_exceptions=True
        )
        
        # Handle exceptions from any client and log results
        ser_signals_list = []
        fer_signals_list = []
        vitals_signals_list = []
        
        if isinstance(ser_signals, Exception):
            logger.warning(f"SER client raised exception: {ser_signals}")
            ser_signals = []
        else:
            ser_signals_list = [{"emotion_label": s.emotion_label, "confidence": s.confidence, "timestamp": s.timestamp} for s in ser_signals]
            logger.info(f"SER returned {len(ser_signals)} signals: {ser_signals_list}")
        
        if isinstance(fer_signals, Exception):
            logger.warning(f"FER client raised exception: {fer_signals}")
            fer_signals = []
        else:
            fer_signals_list = [{"emotion_label": s.emotion_label, "confidence": s.confidence, "timestamp": s.timestamp} for s in fer_signals]
            logger.info(f"FER returned {len(fer_signals)} signals: {fer_signals_list}")
        
        if isinstance(vitals_signals, Exception):
            logger.warning(f"Vitals client raised exception: {vitals_signals}")
            vitals_signals = []
        else:
            vitals_signals_list = [{"emotion_label": s.emotion_label, "confidence": s.confidence, "timestamp": s.timestamp} for s in vitals_signals]
            logger.info(f"Vitals returned {len(vitals_signals)} signals: {vitals_signals_list}")
        
        # Combine all signals
        all_signals: List[ModelSignal] = []
        all_signals.extend(ser_signals)
        all_signals.extend(fer_signals)
        all_signals.extend(vitals_signals)
        
        logger.info(f"Collected {len(all_signals)} total signals "
                   f"(SER: {len(ser_signals)}, FER: {len(fer_signals)}, Vitals: {len(vitals_signals)})")
        
        # Step 3: Filter signals by time window (already done by model clients, but double-check)
        # The model clients should have already filtered, but we'll validate timestamps are within window
        filtered_signals = []
        window_start = snapshot_timestamp.timestamp() - window_seconds
        
        for signal in all_signals:
            try:
                signal_timestamp = datetime.fromisoformat(signal.timestamp.replace('Z', '+00:00'))
                if signal_timestamp.tzinfo is None:
                    signal_timestamp = signal_timestamp.replace(tzinfo=database.get_malaysia_timezone())
                else:
                    signal_timestamp = signal_timestamp.astimezone(database.get_malaysia_timezone())
                
                signal_ts = signal_timestamp.timestamp()
                if signal_ts >= window_start:
                    filtered_signals.append(signal)
                else:
                    logger.debug(f"Filtered out stale signal: {signal.timestamp} (outside {window_seconds}s window)")
            except Exception as e:
                logger.warning(f"Failed to parse signal timestamp {signal.timestamp}: {e}")
                # Include it anyway (model clients should have validated)
                filtered_signals.append(signal)
        
        logger.debug(f"After time window filtering: {len(filtered_signals)} signals")
        
        # Step 4: Check minimum signals requirement
        if not filtered_signals:
            logger.warning("No valid signals found within time window")
            fusion_duration = (datetime.now() - fusion_start_time).total_seconds()
            activity_logger.log_fusion_activity(
                user_id=request.user_id,
                timestamp=snapshot_timestamp,
                status="no_signals",
                ser_signals_count=len(ser_signals),
                fer_signals_count=len(fer_signals),
                vitals_signals_count=len(vitals_signals),
                ser_signals=ser_signals_list,
                fer_signals=fer_signals_list,
                vitals_signals=vitals_signals_list,
                fusion_calculation_log="No signals after filtering",
                duration_seconds=fusion_duration
            )
            return NoSignalsResponse(reason="no valid modality outputs")
        
        # Step 5: Run fusion logic
        logger.info(f"Running fusion logic on {len(filtered_signals)} filtered signals...")
        fusion_calculation_log = f"Input signals: {len(filtered_signals)} total"
        try:
            fused_result = fuse_signals(filtered_signals)
            fusion_calculation_log += f" | Result: {fused_result['emotion_label']} (confidence: {fused_result['confidence_score']:.3f}, emotional_score: {fused_result['emotional_score']})"
            fusion_calculation_log += f" | Signals used: {len(fused_result.get('signals_used', []))}"
            logger.info(f"Fusion calculation complete: {fused_result['emotion_label']} (confidence: {fused_result['confidence_score']:.3f})")
        except ValueError as e:
            logger.error(f"Fusion calculation failed: {e}")
            fusion_calculation_log += f" | Error: {str(e)}"
            fusion_duration = (datetime.now() - fusion_start_time).total_seconds()
            activity_logger.log_fusion_activity(
                user_id=request.user_id,
                timestamp=snapshot_timestamp,
                status="error",
                ser_signals_count=len(ser_signals),
                fer_signals_count=len(fer_signals),
                vitals_signals_count=len(vitals_signals),
                ser_signals=ser_signals_list,
                fer_signals=fer_signals_list,
                vitals_signals=vitals_signals_list,
                fusion_calculation_log=fusion_calculation_log,
                error=str(e),
                duration_seconds=fusion_duration
            )
            return NoSignalsResponse(reason=str(e))
        
        # Step 6: Write to database
        logger.info("Writing fused result to database...")
        db_write_success = False
        inserted_id = database.insert_emotional_log(
            user_id=request.user_id,
            timestamp=snapshot_timestamp,
            emotion_label=fused_result["emotion_label"],
            confidence_score=fused_result["confidence_score"],
            emotional_score=fused_result["emotional_score"]
        )
        
        if inserted_id is None:
            logger.error("Failed to insert emotion log to database")
            db_write_success = False
        else:
            logger.info(f"Successfully wrote to database (ID: {inserted_id})")
            db_write_success = True
        
        # Step 7: Log activity
        fusion_duration = (datetime.now() - fusion_start_time).total_seconds()
        activity_logger.log_fusion_activity(
            user_id=request.user_id,
            timestamp=snapshot_timestamp,
            status="success",
            emotion_label=fused_result["emotion_label"],
            confidence_score=fused_result["confidence_score"],
            signals_used=[{"modality": sig["modality"], "emotion_label": sig["emotion_label"], "confidence": sig["confidence"]} 
                         for sig in fused_result["signals_used"]],
            ser_signals_count=len(ser_signals),
            fer_signals_count=len(fer_signals),
            vitals_signals_count=len(vitals_signals),
            ser_signals=ser_signals_list,
            fer_signals=fer_signals_list,
            vitals_signals=vitals_signals_list,
            db_write_success=db_write_success,
            fusion_calculation_log=fusion_calculation_log,
            duration_seconds=fusion_duration
        )
        
        # Step 8: Return response
        response = FusedEmotionResponse(
            user_id=request.user_id,
            timestamp=snapshot_timestamp.isoformat(),
            emotion_label=fused_result["emotion_label"],
            confidence_score=fused_result["confidence_score"],
            emotional_score=fused_result["emotional_score"],
            signals_used=[
                SignalUsed(**sig) for sig in fused_result["signals_used"]
            ]
        )
        
        logger.info(f"Successfully processed emotion snapshot for user {request.user_id}: "
                   f"{response.emotion_label} (confidence: {response.confidence_score:.3f})")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error processing emotion snapshot: {e}")
        # Log error activity
        try:
            snapshot_timestamp = database.get_current_time_utc8()
            activity_logger.log_fusion_activity(
                user_id=request.user_id,
                timestamp=snapshot_timestamp,
                status="error",
                error=f"Validation error: {str(e)}"
            )
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Error processing emotion snapshot for user {request.user_id}: {e}", exc_info=True)
        # Log error activity
        try:
            snapshot_timestamp = database.get_current_time_utc8()
            activity_logger.log_fusion_activity(
                user_id=request.user_id,
                timestamp=snapshot_timestamp,
                status="error",
                error=f"Exception: {str(e)}"
            )
        except Exception:
            pass
        raise


def parse_demo_signal_string(input_str: str, user_id: str, modality: str, snapshot_timestamp: datetime) -> List[ModelSignal]:
    """
    Parse simplified signal string format into ModelSignal objects.
    
    Format: "emotion:confidence" or "emotion1:conf1,emotion2:conf2"
    Examples: "Sad:0.8" or "Sad:0.8,Happy:0.6"
    
    Args:
        input_str: Input string (e.g., "Sad:0.82,Happy:0.60")
        user_id: User ID
        modality: Modality name ("speech", "face", "vitals")
        snapshot_timestamp: Timestamp to use for signals
    
    Returns:
        List of ModelSignal objects
    """
    signals = []
    
    if not input_str or input_str.strip() == "":
        return signals
    
    # Split by comma for multiple signals
    signal_strings = [s.strip() for s in input_str.split(",")]
    
    malaysia_tz = database.get_malaysia_timezone()
    
    # Emotion label mapping (case-insensitive)
    emotion_mapping = {
        "angry": "Angry",
        "sad": "Sad",
        "happy": "Happy",
        "fear": "Fear"
    }
    
    for signal_str in signal_strings:
        parts = signal_str.split(":")
        
        if len(parts) < 2:
            logger.warning(f"Invalid format: {signal_str} (expected emotion:confidence)")
            continue
        
        emotion_label_raw = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except ValueError:
            logger.warning(f"Invalid confidence: {parts[1]}")
            continue
        
        # Normalize emotion label (case-insensitive)
        emotion_label_lower = emotion_label_raw.lower()
        emotion_label = emotion_mapping.get(emotion_label_lower)
        
        if emotion_label is None:
            valid_emotions = ["Angry", "Sad", "Happy", "Fear"]
            logger.warning(f"Invalid emotion: {emotion_label_raw} (must be one of {valid_emotions})")
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
            except Exception:
                logger.warning(f"Invalid timestamp: {parts[2]}, using snapshot timestamp")
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


def process_emotion_snapshot_demo(request: EmotionSnapshotDemoRequest) -> Union[FusedEmotionResponse, NoSignalsResponse]:
    """
    Process a demo emotion snapshot request with direct signal input.
    
    This bypasses model service calls and uses signals provided directly in the request.
    Used for testing and demonstrations only.
    
    Args:
        request: EmotionSnapshotDemoRequest with user_id and signals dict
    
    Returns:
        FusedEmotionResponse with fused result, or NoSignalsResponse if no valid signals
    """
    logger.info(f"Processing demo emotion snapshot request for user {request.user_id}")
    
    try:
        # Step 1: Validate request
        import uuid
        try:
            uuid.UUID(request.user_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid user_id format: '{request.user_id}'. Must be a valid UUID.")
        
        # Step 2: Determine snapshot timestamp
        if request.timestamp:
            snapshot_timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
            malaysia_tz = database.get_malaysia_timezone()
            if snapshot_timestamp.tzinfo is None:
                snapshot_timestamp = snapshot_timestamp.replace(tzinfo=malaysia_tz)
            else:
                snapshot_timestamp = snapshot_timestamp.astimezone(malaysia_tz)
        else:
            snapshot_timestamp = database.get_current_time_utc8()
        
        logger.info(f"Demo snapshot timestamp: {snapshot_timestamp.isoformat()}")
        
        # Step 3: Parse signals from request
        all_signals: List[ModelSignal] = []
        
        # Map modality names
        modality_mapping = {
            "speech": "speech",
            "ser": "speech",  # Allow SER as alias
            "face": "face",
            "fer": "face",  # Allow FER as alias
            "vitals": "vitals"
        }
        
        for modality_key, signal_string in request.signals.items():
            # Normalize modality key
            modality_key_lower = modality_key.lower()
            modality = modality_mapping.get(modality_key_lower)
            
            if not modality:
                logger.warning(f"Unknown modality key: {modality_key}, skipping")
                continue
            
            if not signal_string or signal_string.strip() == "":
                continue
            
            # Parse signal string
            parsed_signals = parse_demo_signal_string(
                signal_string,
                request.user_id,
                modality,
                snapshot_timestamp
            )
            all_signals.extend(parsed_signals)
            logger.debug(f"Parsed {len(parsed_signals)} signals for {modality}")
        
        logger.info(f"Total signals parsed: {len(all_signals)}")
        
        # Step 4: Check minimum signals requirement
        if not all_signals:
            logger.warning("No valid signals found in demo request")
            return NoSignalsResponse(reason="no valid signals provided")
        
        # Step 5: Run fusion logic
        logger.debug("Running fusion logic on demo signals...")
        try:
            fused_result = fuse_signals(all_signals)
        except ValueError as e:
            logger.error(f"Fusion failed: {e}")
            return NoSignalsResponse(reason=str(e))
        
        # Step 6: Write to database
        logger.debug("Writing fused result to database...")
        inserted_id = database.insert_emotional_log(
            user_id=request.user_id,
            timestamp=snapshot_timestamp,
            emotion_label=fused_result["emotion_label"],
            confidence_score=fused_result["confidence_score"],
            emotional_score=fused_result["emotional_score"]
        )
        
        if inserted_id is None:
            logger.error("Failed to insert emotion log to database")
            # Still return the fused result even if DB write fails
        
        # Step 7: Return response
        response = FusedEmotionResponse(
            user_id=request.user_id,
            timestamp=snapshot_timestamp.isoformat(),
            emotion_label=fused_result["emotion_label"],
            confidence_score=fused_result["confidence_score"],
            emotional_score=fused_result["emotional_score"],
            signals_used=[
                SignalUsed(**sig) for sig in fused_result["signals_used"]
            ]
        )
        
        logger.info(f"Successfully processed demo emotion snapshot for user {request.user_id}: "
                   f"{response.emotion_label} (confidence: {response.confidence_score:.3f})")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error processing demo emotion snapshot: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing demo emotion snapshot for user {request.user_id}: {e}", exc_info=True)
        raise

