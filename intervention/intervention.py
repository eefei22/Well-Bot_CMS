"""
Intervention Service Orchestrator

This module orchestrates the complete intervention suggestion flow:
- Accepts requests
- Calls fusion service to get latest emotion snapshot
- Fetches data from database
- Calls decision engine and suggestion engine
- Returns structured response
"""

from typing import Optional
import logging
import os
import httpx
from datetime import datetime
import uuid

from utils import database
from utils import activity_logger
from intervention.decision_engine import decide_trigger_intervention
from intervention.suggestion_engine import suggest_activities
from intervention.models import (
    SuggestionRequest,
    SuggestionResponse,
    DecisionResult,
    SuggestionResult,
    RankedActivity
)

logger = logging.getLogger(__name__)

# Valid emotion labels
VALID_EMOTION_LABELS = ['Sad', 'Angry', 'Happy', 'Fear']


async def call_fusion_service(user_id: str) -> Optional[dict]:
    """
    Call fusion service to get latest emotion snapshot.
    
    This triggers fusion to query SER, FER, and Vitals services and write
    the fused result to the emotional_log table.
    
    Args:
        user_id: User UUID
        
    Returns:
        Dictionary with fusion result, or None if fusion call failed
    """
    fusion_url = os.getenv("FUSION_SERVICE_URL", "http://localhost:8000")
    fusion_endpoint = f"{fusion_url}/emotion/snapshot"
    
    payload = {
        "user_id": user_id
    }
    
    try:
        logger.info(f"Calling fusion service at {fusion_endpoint} for user {user_id}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                fusion_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if fusion returned "no_signals" response
            if result.get("status") == "no_signals":
                logger.warning(f"Fusion service returned no signals: {result.get('reason', 'unknown')}")
                return None
            
            logger.info(
                f"Fusion completed: {result.get('emotion_label', 'unknown')} "
                f"(confidence: {result.get('confidence_score', 0.0):.2f})"
            )
            return result
            
    except httpx.TimeoutException:
        logger.warning(f"Fusion service call timed out after 30s")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Fusion service returned HTTP {e.response.status_code}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Fusion service call failed: {e}", exc_info=True)
        return None


def validate_suggestion_request(request: SuggestionRequest) -> None:
    """
    Validate suggestion request input parameters.
    
    Args:
        request: SuggestionRequest to validate
    
    Raises:
        ValueError: If validation fails with clear error message
    """
    # Validate user_id is a valid UUID
    try:
        uuid.UUID(request.user_id)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid user_id format: '{request.user_id}'. Must be a valid UUID.")


async def process_suggestion_request(request: SuggestionRequest) -> SuggestionResponse:
    """
    Process an intervention suggestion request.
    
    This orchestrates the complete flow:
    1. Call fusion service to get latest emotion snapshot (queries SER, FER, Vitals)
    2. Fetch latest emotion from database (written by fusion)
    3. Fetch user data from database (emotion logs, preferences, activity counts)
    4. Calculate time since last activity
    5. Call decision engine for kick-start decision
    6. Call suggestion engine for activity recommendations (with frequency-based multipliers)
    7. Return structured response
    
    Args:
        request: SuggestionRequest with user_id
    
    Returns:
        SuggestionResponse with decision and suggestion results
    """
    logger.info(f"Processing suggestion request for user {request.user_id}")
    
    intervention_start_time = datetime.now()
    fusion_called = False
    fusion_status = None
    
    try:
        # Validate request input
        validate_suggestion_request(request)
        
        # Step 1: Call fusion service first to get latest emotion snapshot
        logger.debug("Calling fusion service for latest emotion snapshot...")
        fusion_result = await call_fusion_service(request.user_id)
        fusion_called = True
        if fusion_result:
            fusion_status = "success"
            logger.info(
                f"Fusion completed: {fusion_result.get('emotion_label', 'unknown')} "
                f"(confidence: {fusion_result.get('confidence_score', 0.0):.2f})"
            )
        else:
            fusion_status = "failed"
            logger.warning("Fusion service call failed or returned no signals. Continuing with database lookup...")
        
        # Step 2: Fetch latest emotion from database (written by fusion)
        logger.debug("Fetching latest emotion from database...")
        latest_emotion = database.get_latest_emotion_log(request.user_id)
        if not latest_emotion:
            raise ValueError(f"No emotion logs found for user {request.user_id}")
        
        emotion_label = latest_emotion.get('emotion_label')
        confidence_score = latest_emotion.get('confidence_score')
        emotion_timestamp_str = latest_emotion.get('timestamp')
        
        if not emotion_label or confidence_score is None or not emotion_timestamp_str:
            raise ValueError(f"Invalid emotion log data for user {request.user_id}: missing required fields")
        
        # Validate emotion_label
        if emotion_label not in VALID_EMOTION_LABELS:
            raise ValueError(f"Invalid emotion_label: '{emotion_label}'. Must be one of: {', '.join(VALID_EMOTION_LABELS)}")
        
        # Parse timestamp string to datetime if needed (database stores in UTC+8)
        if isinstance(emotion_timestamp_str, str):
            emotion_timestamp = database.parse_database_timestamp(emotion_timestamp_str)
        else:
            emotion_timestamp = emotion_timestamp_str
        
        logger.info(f"Using latest emotion from database: {emotion_label} (confidence: {confidence_score:.2f})")
        
        # Step 3: Fetch other user data from database
        logger.debug("Fetching other user data from database...")
        recent_emotion_logs = database.fetch_recent_emotion_logs(request.user_id, hours=48)
        user_preferences = database.fetch_user_preferences(request.user_id)
        time_since_last_activity = database.get_time_since_last_activity(request.user_id)
        activity_counts = database.get_activity_counts(request.user_id, days=30)
        
        logger.debug(f"Fetched {len(recent_emotion_logs)} emotion logs")
        logger.debug(f"Activity counts (last 30 days): {activity_counts}")
        
        # Step 4: Call decision engine with fetched emotion
        logger.debug("Calling decision engine...")
        trigger_intervention, decision_confidence, decision_reasoning = decide_trigger_intervention(
            emotion_label=emotion_label,
            confidence_score=confidence_score,
            time_since_last_activity_minutes=time_since_last_activity
        )
        
        decision_result = DecisionResult(
            trigger_intervention=trigger_intervention,
            confidence_score=decision_confidence,
            reasoning=decision_reasoning
        )
        
        # Step 5: Call suggestion engine with fetched emotion (always generate suggestions, regardless of trigger decision)
        logger.debug("Calling suggestion engine...")
        ranked_activities_list, suggestion_reasoning = suggest_activities(
            emotion_label=emotion_label,
            user_preferences=user_preferences,
            activity_counts=activity_counts
        )
        
        # Convert to RankedActivity models
        ranked_activities = [
            RankedActivity(
                activity_type=item['activity_type'],
                rank=item['rank'],
                score=item['score']
            )
            for item in ranked_activities_list
        ]
        
        suggestion_result = SuggestionResult(
            ranked_activities=ranked_activities,
            reasoning=suggestion_reasoning
        )
        
        # Step 6: Build and return response
        response = SuggestionResponse(
            user_id=request.user_id,
            decision=decision_result,
            suggestion=suggestion_result
        )
        
        # Log successful activity
        intervention_duration = (datetime.now() - intervention_start_time).total_seconds()
        activity_logger.log_intervention_activity(
            user_id=request.user_id,
            timestamp=intervention_start_time,
            status="success",
            trigger_intervention=decision_result.trigger_intervention,
            decision_confidence=decision_result.confidence_score,
            decision_reasoning=decision_result.reasoning,
            emotion_label=emotion_label,
            emotion_confidence=confidence_score,
            ranked_activities=[
                {
                    "activity_type": activity.activity_type,
                    "rank": activity.rank,
                    "score": activity.score
                }
                for activity in ranked_activities
            ],
            fusion_called=fusion_called,
            fusion_status=fusion_status,
            duration_seconds=intervention_duration
        )
        
        logger.info(f"Successfully processed suggestion request for user {request.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing suggestion request for user {request.user_id}: {e}", exc_info=True)
        # Log error activity
        intervention_duration = (datetime.now() - intervention_start_time).total_seconds()
        activity_logger.log_intervention_activity(
            user_id=request.user_id,
            timestamp=intervention_start_time,
            status="error",
            fusion_called=fusion_called,
            fusion_status=fusion_status,
            error=str(e),
            duration_seconds=intervention_duration
        )
        raise

