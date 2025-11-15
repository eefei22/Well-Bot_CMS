"""
Intervention Service Orchestrator

This module orchestrates the complete intervention suggestion flow:
- Accepts requests
- Fetches data from database
- Calls decision engine and suggestion engine
- Returns structured response
"""

from typing import Optional
import logging
from datetime import datetime

from utils import database
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


def get_time_of_day_context(timestamp: datetime) -> str:
    """
    Derive time of day context from timestamp using Malaysian timezone (UTC+8).
    
    Time periods:
    - morning: 5:00 - 11:59
    - afternoon: 12:00 - 16:59
    - evening: 17:00 - 20:59
    - night: 21:00 - 4:59
    
    Args:
        timestamp: Datetime object (assumed UTC if timezone-naive)
    
    Returns:
        One of: 'morning', 'afternoon', 'evening', 'night'
    """
    from datetime import timezone, timedelta
    
    # Try zoneinfo first (requires tzdata package on Windows)
    MALAYSIA_TZ = None
    try:
        from zoneinfo import ZoneInfo
        MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")
    except (ImportError, Exception):
        # ZoneInfoNotFoundError, ImportError, or other issues - fall back to pytz
        try:
            import pytz
            MALAYSIA_TZ = pytz.timezone("Asia/Kuala_Lumpur")
        except ImportError:
            # Final fallback: manual UTC+8 offset
            MALAYSIA_TZ = timezone(timedelta(hours=8))
    
    # Convert to Malaysian timezone
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    timestamp_malaysia = timestamp.astimezone(MALAYSIA_TZ)
    hour = timestamp_malaysia.hour
    
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:  # 21 <= hour < 5
        return 'night'


def process_suggestion_request(request: SuggestionRequest) -> SuggestionResponse:
    """
    Process an intervention suggestion request.
    
    This orchestrates the complete flow:
    1. Fetch latest emotion from database
    2. Fetch user data from database (emotion logs, activity logs, preferences)
    3. Calculate time since last activity
    4. Call decision engine for kick-start decision
    5. Call suggestion engine for activity recommendations
    6. Return structured response
    
    Args:
        request: SuggestionRequest with user_id
    
    Returns:
        SuggestionResponse with decision and suggestion results
    """
    logger.info(f"Processing suggestion request for user {request.user_id}")
    
    try:
        # 1. Fetch latest emotion from database
        logger.debug("Fetching latest emotion from database...")
        latest_emotion = database.get_latest_emotion_log(request.user_id)
        if not latest_emotion:
            raise ValueError(f"No emotion logs found for user {request.user_id}")
        
        emotion_label = latest_emotion.get('emotion_label')
        confidence_score = latest_emotion.get('confidence_score')
        emotion_timestamp_str = latest_emotion.get('timestamp')
        
        if not emotion_label or confidence_score is None or not emotion_timestamp_str:
            raise ValueError(f"Invalid emotion log data for user {request.user_id}: missing required fields")
        
        # Parse timestamp string to datetime if needed
        if isinstance(emotion_timestamp_str, str):
            emotion_timestamp = datetime.fromisoformat(emotion_timestamp_str.replace('Z', '+00:00'))
        else:
            emotion_timestamp = emotion_timestamp_str
        
        logger.info(f"Using latest emotion from database: {emotion_label} (confidence: {confidence_score:.2f})")
        
        # 2. Fetch other user data from database
        logger.debug("Fetching other user data from database...")
        recent_emotion_logs = database.fetch_recent_emotion_logs(request.user_id, hours=48)
        recent_activity_logs = database.fetch_recent_activity_logs(request.user_id, hours=24)
        user_preferences = database.fetch_user_preferences(request.user_id)
        time_since_last_activity = database.get_time_since_last_activity(request.user_id)
        
        logger.debug(f"Fetched {len(recent_emotion_logs)} emotion logs, {len(recent_activity_logs)} activity logs")
        
        # 3. Determine time of day context
        time_of_day = request.context_time_of_day
        if not time_of_day:
            time_of_day = get_time_of_day_context(emotion_timestamp)
        
        logger.debug(f"Time of day context: {time_of_day}")
        
        # 4. Call decision engine with fetched emotion
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
        
        # 5. Call suggestion engine with fetched emotion (always generate suggestions, regardless of trigger decision)
        logger.debug("Calling suggestion engine...")
        ranked_activities_list, suggestion_reasoning = suggest_activities(
            emotion_label=emotion_label,
            user_preferences=user_preferences,
            recent_activity_logs=recent_activity_logs,
            time_of_day=time_of_day
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
        
        # 5. Build and return response
        response = SuggestionResponse(
            user_id=request.user_id,
            decision=decision_result,
            suggestion=suggestion_result
        )
        
        logger.info(f"Successfully processed suggestion request for user {request.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing suggestion request for user {request.user_id}: {e}", exc_info=True)
        raise

