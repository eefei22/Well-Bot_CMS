"""
Suggestion Engine

This module implements the activity recommendation algorithm to determine
which activities to suggest and in what order.
"""

from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Available activity types
ACTIVITY_TYPES = ['journal', 'gratitude', 'meditation', 'quote']

# Base emotion-to-activity mapping weights (0.0 to 1.0)
# Higher weight = better match for that emotion
EMOTION_ACTIVITY_WEIGHTS = {
    'Sad': {
        'journal': 0.9,
        'meditation': 0.8,
        'gratitude': 0.7,
        'quote': 0.6
    },
    'Angry': {
        'meditation': 0.9,
        'journal': 0.7,
        'quote': 0.6,
        'gratitude': 0.5
    },
    'Fear': {
        'gratitude': 0.8,
        'journal': 0.7,
        'meditation': 0.7,
        'quote': 0.6
    },
    'Happy': {
        'gratitude': 0.8,
        'journal': 0.7,
        'quote': 0.6,
        'meditation': 0.5
    }
}

# Time-of-day adjustments (multipliers)
TIME_OF_DAY_ADJUSTMENTS = {
    'morning': {
        'journal': 0.8,
        'gratitude': 0.9,
        'meditation': 0.7,
        'quote': 0.8
    },
    'afternoon': {
        'journal': 0.7,
        'gratitude': 0.8,
        'meditation': 0.8,
        'quote': 0.9
    },
    'evening': {
        'journal': 0.9,
        'meditation': 0.9,
        'gratitude': 0.7,
        'quote': 0.8
    },
    'night': {
        'journal': 0.9,
        'meditation': 0.8,
        'gratitude': 0.6,
        'quote': 0.7
    }
}

# Preference field mapping (from users.prefer_intervention to activity types)
PREFERENCE_MAPPING = {
    'journaling': 'journal',
    'gratitude': 'gratitude',
    'breathing': 'meditation',  # meditation includes breathing
    'quote': 'quote'
}


def suggest_activities(
    emotion_label: str,
    user_preferences: Dict,
    recent_activity_logs: List[Dict],
    time_of_day: Optional[str] = None
) -> Tuple[List[Dict], Optional[str]]:
    """
    Suggest activities ranked 1-5 with scores and reasoning.
    
    Args:
        emotion_label: Current emotion label ('Angry', 'Sad', 'Happy', 'Fear')
        user_preferences: Dictionary from users.prefer_intervention JSONB field
        recent_activity_logs: List of recent activity log dictionaries
        time_of_day: Optional time of day context ('morning', 'afternoon', 'evening', 'night')
    
    Returns:
        Tuple of (ranked_activities: List[Dict], reasoning: Optional[str])
        - ranked_activities: List of dicts with 'activity_type', 'rank' (1-5), 'score' (0.0-1.0)
        - reasoning: Optional string explaining the suggestions
    """
    # Initialize base scores from emotion mapping
    emotion_weights = EMOTION_ACTIVITY_WEIGHTS.get(emotion_label, {
        'journal': 0.7,
        'gratitude': 0.7,
        'meditation': 0.7,
        'quote': 0.7
    })
    
    # Start with base emotion weights
    activity_scores = {activity: emotion_weights.get(activity, 0.5) for activity in ACTIVITY_TYPES}
    
    # Apply user preference adjustments
    for pref_key, pref_value in user_preferences.items():
        activity_type = PREFERENCE_MAPPING.get(pref_key)
        if activity_type and activity_type in activity_scores:
            if pref_value:  # User prefers this activity
                activity_scores[activity_type] *= 1.2  # Boost by 20%
            else:  # User doesn't prefer this activity
                activity_scores[activity_type] *= 0.7  # Reduce by 30%
    
    # Apply time-of-day adjustments
    if time_of_day and time_of_day in TIME_OF_DAY_ADJUSTMENTS:
        time_adjustments = TIME_OF_DAY_ADJUSTMENTS[time_of_day]
        for activity in activity_scores:
            if activity in time_adjustments:
                activity_scores[activity] *= time_adjustments[activity]
    
    # Apply recent activity penalty (penalize recently used activities)
    recent_activity_types = [log.get('intervention_type') for log in recent_activity_logs[-5:]]  # Last 5 activities
    for activity_type in recent_activity_types:
        if activity_type in activity_scores:
            activity_scores[activity_type] *= 0.8  # Reduce by 20% if recently used
    
    # Normalize scores to 0.0-1.0 range
    max_score = max(activity_scores.values()) if activity_scores.values() else 1.0
    if max_score > 0:
        activity_scores = {k: min(v / max_score, 1.0) for k, v in activity_scores.items()}
    
    # Sort activities by score (descending)
    sorted_activities = sorted(
        activity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create ranked list (1-5, where 1 is best)
    ranked_activities = []
    for rank, (activity_type, score) in enumerate(sorted_activities, start=1):
        ranked_activities.append({
            'activity_type': activity_type,
            'rank': rank,
            'score': round(score, 3)
        })
    
    # Generate reasoning
    reasoning_parts = []
    reasoning_parts.append(f"Emotion: {emotion_label}")
    if time_of_day:
        reasoning_parts.append(f"Time of day: {time_of_day}")
    top_activity = ranked_activities[0] if ranked_activities else None
    if top_activity:
        reasoning_parts.append(f"Top suggestion: {top_activity['activity_type']} (score: {top_activity['score']:.3f})")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else None
    
    # Log suggested activities
    activity_summary = ", ".join([f"{a['activity_type']} (rank {a['rank']}, score {a['score']:.3f})" for a in ranked_activities])
    logger.info(f"Suggested activities: {activity_summary}")
    
    return ranked_activities, reasoning

