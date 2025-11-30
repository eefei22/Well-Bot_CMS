"""
Suggestion Engine

This module implements the activity recommendation algorithm to determine
which activities to suggest and in what order.
"""

from typing import List, Dict, Optional, Tuple
import logging

from intervention.config_loader import load_config

logger = logging.getLogger(__name__)

# Load configuration
_config = load_config()
_suggestion_config = _config.get("suggestion_engine", {})

# Available activity types (from config with fallback)
ACTIVITY_TYPES = _suggestion_config.get("activity_types", ['journal', 'gratitude', 'meditation', 'quote'])

# Base emotion-to-activity mapping weights (0.0 to 1.0)
# Higher weight = better match for that emotion (from config with fallback)
_emotion_weights_config = _suggestion_config.get("emotion_activity_weights", {})
EMOTION_ACTIVITY_WEIGHTS = {
    'Sad': _emotion_weights_config.get('Sad', {'journal': 0.9, 'meditation': 0.8, 'gratitude': 0.7, 'quote': 0.6}),
    'Angry': _emotion_weights_config.get('Angry', {'meditation': 0.9, 'journal': 0.7, 'quote': 0.6, 'gratitude': 0.5}),
    'Fear': _emotion_weights_config.get('Fear', {'meditation': 0.8, 'quote': 0.7, 'journal': 0.7, 'gratitude': 0.6}),
    'Happy': _emotion_weights_config.get('Happy', {'gratitude': 0.8, 'journal': 0.7, 'quote': 0.6, 'meditation': 0.5})
}

# Frequency-based multipliers (based on relative usage frequency)
# Most frequent activity gets highest multiplier, least frequent gets lowest (from config with fallback)
_frequency_multipliers_config = _suggestion_config.get("frequency_multipliers", {})
FREQUENCY_MULTIPLIERS = {
    1: _frequency_multipliers_config.get("1", 1.3),
    2: _frequency_multipliers_config.get("2", 1.2),
    3: _frequency_multipliers_config.get("3", 1.1),
    4: _frequency_multipliers_config.get("4", 1.05)
}

# Preference multipliers (from config with fallback)
_preference_multipliers_config = _suggestion_config.get("preference_multipliers", {})
PREFERRED_MULTIPLIER = _preference_multipliers_config.get("preferred", 1.2)
NOT_PREFERRED_MULTIPLIER = _preference_multipliers_config.get("not_preferred", 0.7)

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
    activity_counts: Dict[str, int]
) -> Tuple[List[Dict], Optional[str]]:
    """
    Suggest activities ranked 1-4 with scores and reasoning.
    
    Args:
        emotion_label: Current emotion label ('Angry', 'Sad', 'Happy', 'Fear')
        user_preferences: Dictionary from users.prefer_intervention JSONB field
        activity_counts: Dictionary with activity type as key and count as value
                         Example: {'journal': 15, 'gratitude': 8, 'meditation': 12, 'quote': 5}
    
    Returns:
        Tuple of (ranked_activities: List[Dict], reasoning: Optional[str])
        - ranked_activities: List of dicts with 'activity_type', 'rank' (1-4), 'score' (0.0-1.0)
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
                activity_scores[activity_type] *= PREFERRED_MULTIPLIER
            else:  # User doesn't prefer this activity
                activity_scores[activity_type] *= NOT_PREFERRED_MULTIPLIER
    
    # Apply frequency-based multipliers
    # Get counts for all activity types (default to 0 if not in dict)
    counts = {activity: activity_counts.get(activity, 0) for activity in ACTIVITY_TYPES}
    
    # Group activities by their frequency count
    # Activities with the same count will get the same multiplier
    frequency_groups = {}
    for activity in ACTIVITY_TYPES:
        count = counts[activity]
        if count not in frequency_groups:
            frequency_groups[count] = []
        frequency_groups[count].append(activity)
    
    # Sort groups by count (descending) to determine group rank
    sorted_group_counts = sorted(frequency_groups.keys(), reverse=True)
    
    # Assign multipliers to groups and apply to activities
    # All activities in the same group get the same multiplier
    for group_rank, group_count in enumerate(sorted_group_counts, start=1):
        multiplier = FREQUENCY_MULTIPLIERS.get(group_rank, 1.0)
        activities_in_group = frequency_groups[group_count]
        
        for activity_type in activities_in_group:
            if activity_type in activity_scores:
                activity_scores[activity_type] *= multiplier
                logger.debug(f"Applied frequency multiplier {multiplier}x to {activity_type} (group rank {group_rank}, count {group_count})")
    
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
    
    # Create ranked list (1-4, where 1 is best)
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
    top_activity = ranked_activities[0] if ranked_activities else None
    if top_activity:
        reasoning_parts.append(f"Top suggestion: {top_activity['activity_type']} (score: {top_activity['score']:.3f})")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else None
    
    # Log suggested activities
    activity_summary = ", ".join([f"{a['activity_type']} (rank {a['rank']}, score {a['score']:.3f})" for a in ranked_activities])
    logger.info(f"Suggested activities: {activity_summary}")
    
    return ranked_activities, reasoning

