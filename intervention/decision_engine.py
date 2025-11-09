"""
Decision Engine

This module implements the kick-start decision algorithm to determine
whether an intervention should be triggered.
"""

from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Configuration constants
NEGATIVE_EMOTIONS = ['Sad', 'Angry', 'Fear']
CONFIDENCE_THRESHOLD = 0.70
MIN_TIME_SINCE_LAST_ACTIVITY_MINUTES = 60.0


def decide_trigger_intervention(
    emotion_label: str,
    confidence_score: float,
    time_since_last_activity_minutes: float
) -> Tuple[bool, float, Optional[str]]:
    """
    Decide whether to trigger an intervention based on emotion, confidence, and activity history.
    
    Args:
        emotion_label: Current emotion label ('Angry', 'Sad', 'Happy', 'Fear')
        confidence_score: Confidence score for the emotion (0.0 to 1.0)
        time_since_last_activity_minutes: Minutes since last activity (float('inf') if no activities)
    
    Returns:
        Tuple of (trigger_intervention: bool, confidence_score: float, reasoning: Optional[str])
        - trigger_intervention: True if intervention should be triggered
        - confidence_score: Confidence in the decision (0.0 to 1.0)
        - reasoning: Optional string explaining the decision
    """
    # Check if emotion is negative
    is_negative_emotion = emotion_label in NEGATIVE_EMOTIONS
    
    # Check if confidence meets threshold
    meets_confidence_threshold = confidence_score >= CONFIDENCE_THRESHOLD
    
    # Check if enough time has passed since last activity
    enough_time_passed = time_since_last_activity_minutes > MIN_TIME_SINCE_LAST_ACTIVITY_MINUTES
    
    # Decision logic: trigger if all conditions are met
    should_trigger = is_negative_emotion and meets_confidence_threshold and enough_time_passed
    
    # Calculate decision confidence
    # Base confidence on how well conditions are met
    decision_confidence = 0.0
    reasoning_parts = []
    
    if should_trigger:
        # All conditions met - high confidence
        decision_confidence = min(confidence_score, 0.95)  # Cap at 0.95
        reasoning_parts.append(f"Negative emotion '{emotion_label}' detected")
        reasoning_parts.append(f"Confidence {confidence_score:.2f} >= {CONFIDENCE_THRESHOLD}")
        reasoning_parts.append(f"Time since last activity: {time_since_last_activity_minutes:.1f} minutes")
    else:
        # Some condition not met - lower confidence
        if not is_negative_emotion:
            decision_confidence = 0.0
            reasoning_parts.append(f"Emotion '{emotion_label}' is not negative")
        elif not meets_confidence_threshold:
            decision_confidence = confidence_score * 0.5  # Scale down based on confidence
            reasoning_parts.append(f"Confidence {confidence_score:.2f} < {CONFIDENCE_THRESHOLD}")
        elif not enough_time_passed:
            decision_confidence = 0.0
            reasoning_parts.append(f"Recent activity {time_since_last_activity_minutes:.1f} minutes ago")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else None
    
    logger.info(f"Decision: trigger={should_trigger}, confidence={decision_confidence:.2f}, reason={reasoning}")
    
    return should_trigger, decision_confidence, reasoning

