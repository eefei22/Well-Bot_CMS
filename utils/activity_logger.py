"""
Activity Logger Utility

Provides centralized activity logging for fusion, intervention, and context services.
Logs are written to JSONL files for easy parsing and dashboard display.
"""

import json
import os
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Base directory for activity logs
BASE_LOG_DIR = "data/activity_logs"

# Log directories for each service
FUSION_LOG_DIR = os.path.join(BASE_LOG_DIR, "fusion")
INTERVENTION_LOG_DIR = os.path.join(BASE_LOG_DIR, "intervention")
CONTEXT_LOG_DIR = os.path.join(BASE_LOG_DIR, "context")

# Locks for thread-safe file writing
_fusion_lock = threading.Lock()
_intervention_lock = threading.Lock()
_context_lock = threading.Lock()


def _ensure_log_dir(log_dir: str):
    """Ensure log directory exists."""
    os.makedirs(log_dir, exist_ok=True)


def _get_log_file(log_dir: str, prefix: str) -> str:
    """
    Get log file path for today's date.
    
    Args:
        log_dir: Log directory path
        prefix: File prefix (e.g., "fusion", "intervention")
    
    Returns:
        Path to log file
    """
    _ensure_log_dir(log_dir)
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(log_dir, f"{prefix}_activity_{today}.jsonl")


def log_fusion_activity(
    user_id: str,
    timestamp: datetime,
    status: str,  # "success", "no_signals", "error"
    emotion_label: Optional[str] = None,
    confidence_score: Optional[float] = None,
    signals_used: Optional[list] = None,
    ser_signals_count: int = 0,
    fer_signals_count: int = 0,
    vitals_signals_count: int = 0,
    error: Optional[str] = None,
    duration_seconds: Optional[float] = None
):
    """
    Log fusion service activity.
    
    Args:
        user_id: User UUID
        timestamp: Snapshot timestamp
        status: Activity status ("success", "no_signals", "error")
        emotion_label: Fused emotion label (if successful)
        confidence_score: Fused confidence score (if successful)
        signals_used: List of signals used in fusion
        ser_signals_count: Number of SER signals received
        fer_signals_count: Number of FER signals received
        vitals_signals_count: Number of Vitals signals received
        error: Error message (if failed)
        duration_seconds: Processing duration in seconds
    """
    log_file = _get_log_file(FUSION_LOG_DIR, "fusion")
    
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now().isoformat(),
        "user_id": user_id,
        "status": status,
        "emotion_label": emotion_label,
        "confidence_score": confidence_score,
        "signals_used": signals_used or [],
        "model_signals": {
            "ser": ser_signals_count,
            "fer": fer_signals_count,
            "vitals": vitals_signals_count
        },
        "error": error,
        "duration_seconds": duration_seconds
    }
    
    try:
        with _fusion_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        logger.debug(f"Logged fusion activity to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log fusion activity: {e}", exc_info=True)


def log_intervention_activity(
    user_id: str,
    timestamp: datetime,
    status: str,  # "success", "error"
    trigger_intervention: Optional[bool] = None,
    decision_confidence: Optional[float] = None,
    decision_reasoning: Optional[str] = None,
    emotion_label: Optional[str] = None,
    emotion_confidence: Optional[float] = None,
    ranked_activities: Optional[list] = None,
    fusion_called: bool = False,
    fusion_status: Optional[str] = None,
    error: Optional[str] = None,
    duration_seconds: Optional[float] = None
):
    """
    Log intervention service activity.
    
    Args:
        user_id: User UUID
        timestamp: Request timestamp
        status: Activity status ("success", "error")
        trigger_intervention: Whether intervention was triggered
        decision_confidence: Decision confidence score
        decision_reasoning: Decision reasoning text
        emotion_label: Emotion label used
        emotion_confidence: Emotion confidence score
        ranked_activities: List of ranked activity suggestions
        fusion_called: Whether fusion service was called
        fusion_status: Fusion service call status ("success", "failed", "skipped")
        error: Error message (if failed)
        duration_seconds: Processing duration in seconds
    """
    log_file = _get_log_file(INTERVENTION_LOG_DIR, "intervention")
    
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now().isoformat(),
        "user_id": user_id,
        "status": status,
        "decision": {
            "trigger_intervention": trigger_intervention,
            "confidence": decision_confidence,
            "reasoning": decision_reasoning
        },
        "emotion": {
            "label": emotion_label,
            "confidence": emotion_confidence
        },
        "ranked_activities": ranked_activities or [],
        "fusion": {
            "called": fusion_called,
            "status": fusion_status
        },
        "error": error,
        "duration_seconds": duration_seconds
    }
    
    try:
        with _intervention_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        logger.debug(f"Logged intervention activity to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log intervention activity: {e}", exc_info=True)


def log_context_activity(
    user_id: str,
    timestamp: datetime,
    status: str,  # "success", "partial_success", "error"
    conversation_id: Optional[str] = None,
    facts_extracted: bool = False,
    context_extracted: bool = False,
    facts_length: Optional[int] = None,
    context_length: Optional[int] = None,
    messages_processed: Optional[int] = None,
    chunks_created: Optional[int] = None,
    error: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    embed_duration: Optional[float] = None,
    facts_duration: Optional[float] = None,
    context_duration: Optional[float] = None
):
    """
    Log context generation activity.
    
    Args:
        user_id: User UUID
        timestamp: Request timestamp
        status: Activity status ("success", "partial_success", "error")
        conversation_id: Conversation ID (if provided)
        facts_extracted: Whether facts were extracted successfully
        context_extracted: Whether context was extracted successfully
        facts_length: Length of facts summary in characters
        context_length: Length of context summary in characters
        messages_processed: Number of messages processed for embedding
        chunks_created: Number of chunks created
        error: Error message (if failed)
        duration_seconds: Total processing duration in seconds
        embed_duration: Embedding step duration
        facts_duration: Facts extraction duration
        context_duration: Context extraction duration
    """
    log_file = _get_log_file(CONTEXT_LOG_DIR, "context")
    
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now().isoformat(),
        "user_id": user_id,
        "conversation_id": conversation_id,
        "status": status,
        "results": {
            "facts_extracted": facts_extracted,
            "context_extracted": context_extracted,
            "facts_length": facts_length,
            "context_length": context_length
        },
        "embedding": {
            "messages_processed": messages_processed,
            "chunks_created": chunks_created
        },
        "durations": {
            "total": duration_seconds,
            "embedding": embed_duration,
            "facts": facts_duration,
            "context": context_duration
        },
        "error": error
    }
    
    try:
        with _context_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        logger.debug(f"Logged context activity to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log context activity: {e}", exc_info=True)


def read_activity_logs(
    log_dir: str,
    limit: int = 100,
    user_id: Optional[str] = None
) -> list[Dict[str, Any]]:
    """
    Read activity logs from directory.
    
    Args:
        log_dir: Log directory path
        limit: Maximum number of entries to return
        user_id: Optional filter by user_id
    
    Returns:
        List of log entries (newest first)
    """
    if not os.path.exists(log_dir):
        return []
    
    all_entries = []
    
    try:
        # Read all JSONL files in the directory
        for log_file in Path(log_dir).glob("*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            # Filter by user_id if provided
                            if user_id and entry.get("user_id") != user_id:
                                continue
                            all_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error reading log file {log_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        # Handle both ISO format strings and datetime objects
        def get_sort_key(entry):
            ts = entry.get("timestamp", "")
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except Exception:
                    return 0
            elif isinstance(ts, datetime):
                return ts.timestamp()
            return 0
        
        all_entries.sort(key=get_sort_key, reverse=True)
        
        # Return last N entries
        return all_entries[:limit]
        
    except Exception as e:
        logger.error(f"Error reading activity logs from {log_dir}: {e}", exc_info=True)
        return []

