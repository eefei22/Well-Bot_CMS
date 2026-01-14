"""
Activity Logger Utility

Provides centralized activity logging for fusion, intervention, and context services.
Logs are stored in-memory for real-time dashboard monitoring (non-persistent).
"""

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Any

# Import get_malaysia_timezone - use lazy import to avoid circular dependencies
def _get_malaysia_timezone():
    """Lazy import to avoid circular dependencies."""
    from utils.database import get_malaysia_timezone
    return get_malaysia_timezone()

logger = logging.getLogger(__name__)

# Maximum number of entries to keep in memory per service (auto-cleanup via deque maxlen)
MAX_LOG_ENTRIES = 1000

# In-memory storage for activity logs (newest entries automatically replace oldest)
_fusion_activities: deque = deque(maxlen=MAX_LOG_ENTRIES)
_intervention_activities: deque = deque(maxlen=MAX_LOG_ENTRIES)
_context_activities: deque = deque(maxlen=MAX_LOG_ENTRIES)

# Locks for thread-safe operations
_fusion_lock = threading.Lock()
_intervention_lock = threading.Lock()
_context_lock = threading.Lock()


def _get_service_storage(service_name: str):
    """
    Get the appropriate storage deque and lock based on service_name.
    
    Args:
        service_name: Service identifier ("fusion", "intervention", "context")
    
    Returns:
        Tuple of (deque, lock) for the service
    """
    service_name_lower = service_name.lower()
    if service_name_lower == "fusion" or "fusion" in service_name_lower:
        return _fusion_activities, _fusion_lock
    elif service_name_lower == "intervention" or "intervention" in service_name_lower:
        return _intervention_activities, _intervention_lock
    elif service_name_lower == "context" or "context" in service_name_lower:
        return _context_activities, _context_lock
    else:
        # Default to fusion if unknown
        logger.warning(f"Unknown service_name '{service_name}', defaulting to fusion")
        return _fusion_activities, _fusion_lock


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
    duration_seconds: Optional[float] = None,
    ser_signals: Optional[list] = None,
    fer_signals: Optional[list] = None,
    vitals_signals: Optional[list] = None,
    db_write_success: Optional[bool] = None,
    fusion_calculation_log: Optional[str] = None
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
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now(_get_malaysia_timezone()).isoformat(),
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
        "model_signals_detail": {
            "ser": [{"emotion_label": s.get("emotion_label"), "confidence": s.get("confidence"), "timestamp": s.get("timestamp")} 
                   for s in (ser_signals or [])] if ser_signals else [],
            "fer": [{"emotion_label": s.get("emotion_label"), "confidence": s.get("confidence"), "timestamp": s.get("timestamp")} 
                   for s in (fer_signals or [])] if fer_signals else [],
            "vitals": [{"emotion_label": s.get("emotion_label"), "confidence": s.get("confidence"), "timestamp": s.get("timestamp")} 
                      for s in (vitals_signals or [])] if vitals_signals else []
        },
        "db_write_success": db_write_success,
        "fusion_calculation_log": fusion_calculation_log,
        "error": error,
        "duration_seconds": duration_seconds
    }
    
    try:
        with _fusion_lock:
            _fusion_activities.append(log_entry)
        logger.debug(f"Logged fusion activity (in-memory, {len(_fusion_activities)} entries)")
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
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now(_get_malaysia_timezone()).isoformat(),
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
            _intervention_activities.append(log_entry)
        logger.debug(f"Logged intervention activity (in-memory, {len(_intervention_activities)} entries)")
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
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now(_get_malaysia_timezone()).isoformat(),
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
            _context_activities.append(log_entry)
        logger.debug(f"Logged context activity (in-memory, {len(_context_activities)} entries)")
    except Exception as e:
        logger.warning(f"Failed to log context activity: {e}", exc_info=True)


def read_activity_logs(
    service_name: str,
    limit: int = 100,
    user_id: Optional[str] = None
) -> list[Dict[str, Any]]:
    """
    Read activity logs from in-memory storage.
    
    Args:
        service_name: Service identifier ("fusion", "intervention", "context")
        limit: Maximum number of entries to return
        user_id: Optional filter by user_id
    
    Returns:
        List of log entries (newest first, already sorted by insertion order)
    """
    try:
        # Get the appropriate storage deque and lock
        activities, lock = _get_service_storage(service_name)
        
        with lock:
            # Convert deque to list (deque maintains insertion order, newest last)
            # We want newest first, so reverse it
            all_entries = list(activities)
            all_entries.reverse()  # Newest first
        
        # Filter by user_id if provided
        if user_id:
            all_entries = [entry for entry in all_entries if entry.get("user_id") == user_id]
        
        # Return last N entries (already newest first)
        return all_entries[:limit]
        
    except Exception as e:
        logger.error(f"Error reading activity logs for service '{service_name}': {e}", exc_info=True)
        return []

