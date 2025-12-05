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

from fusion.models import EmotionSnapshotRequest, FusedEmotionResponse, NoSignalsResponse, SignalUsed, ModelSignal
from fusion.model_clients import SERClient, FERClient, VitalsClient
from fusion.fusion_logic import fuse_signals
from fusion.config_loader import load_config
from utils import database

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
        
        # Step 2: Call model services in parallel
        logger.debug("Calling model services in parallel...")
        
        # Initialize clients with optional timeout override
        ser_client = SERClient(timeout=timeout_override)
        fer_client = FERClient(timeout=timeout_override)
        vitals_client = VitalsClient(timeout=timeout_override)
        
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
        
        # Handle exceptions from any client
        if isinstance(ser_signals, Exception):
            logger.warning(f"SER client raised exception: {ser_signals}")
            ser_signals = []
        if isinstance(fer_signals, Exception):
            logger.warning(f"FER client raised exception: {fer_signals}")
            fer_signals = []
        if isinstance(vitals_signals, Exception):
            logger.warning(f"Vitals client raised exception: {vitals_signals}")
            vitals_signals = []
        
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
            return NoSignalsResponse(reason="no valid modality outputs")
        
        # Step 5: Run fusion logic
        logger.debug("Running fusion logic...")
        try:
            fused_result = fuse_signals(filtered_signals)
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
        
        logger.info(f"Successfully processed emotion snapshot for user {request.user_id}: "
                   f"{response.emotion_label} (confidence: {response.confidence_score:.3f})")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error processing emotion snapshot: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing emotion snapshot for user {request.user_id}: {e}", exc_info=True)
        raise

