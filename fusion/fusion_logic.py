"""
Core Fusion Logic for Fusion Service

This module implements the weighted fusion algorithm that aggregates
emotion predictions from multiple modalities.
"""

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from fusion.config_loader import load_config
from fusion.models import ModelSignal

logger = logging.getLogger(__name__)

# Load configuration
_config = load_config()
_fusion_config = _config.get("fusion_weights", {})
FUSION_WEIGHTS = {
    "speech": _fusion_config.get("speech", 0.4),
    "face": _fusion_config.get("face", 0.3),
    "vitals": _fusion_config.get("vitals", 0.3)
}

# Negative emotion boost configuration
_boost_config = _config.get("negative_emotion_boost", {})
NEGATIVE_EMOTION_BOOST_ENABLED = _boost_config.get("enabled", True)
NEGATIVE_EMOTION_BOOST_STRENGTH = _boost_config.get("boost_strength", 0.4)

# Valid emotion labels
VALID_EMOTIONS = ["Angry", "Sad", "Happy", "Fear"]

# Negative emotions (critical emotions that require attention)
NEGATIVE_EMOTIONS = ["Angry", "Sad", "Fear"]


def calculate_mood_score(emotion_confidences: Dict[str, float]) -> int:
    """
    Calculate balanced mood score from emotion confidences using balanced average approach.
    
    This computes overall emotional valence (positive vs negative) by:
    1. Averaging negative emotions (Sad, Angry, Fear) to balance against single positive (Happy)
    2. Computing net mood: positive - average_negative
    3. Scaling to 0-100 range (0=very negative, 50=neutral, 100=very positive)
    
    Args:
        emotion_confidences: Dictionary mapping emotion labels to normalized confidence scores (0.0-1.0)
            Keys: "Happy", "Sad", "Angry", "Fear"
            Values: Normalized confidence (float 0.0-1.0)
            Missing emotions are treated as 0.0
    
    Returns:
        Integer mood score (0-100) representing overall emotional valence
    """
    # Extract emotion confidences (default to 0.0 if missing)
    happy_confidence = emotion_confidences.get("Happy", 0.0)
    sad_confidence = emotion_confidences.get("Sad", 0.0)
    anger_confidence = emotion_confidences.get("Angry", 0.0)
    fear_confidence = emotion_confidences.get("Fear", 0.0)
    
    # Build lists of present negative emotions
    negative_values = []
    if sad_confidence > 0.0:
        negative_values.append(sad_confidence)
    if anger_confidence > 0.0:
        negative_values.append(anger_confidence)
    if fear_confidence > 0.0:
        negative_values.append(fear_confidence)
    
    # Handle edge case: no emotions present (all 0.0)
    if happy_confidence == 0.0 and len(negative_values) == 0:
        return 50  # Neutral mood
    
    # Calculate average negative emotion
    if len(negative_values) > 0:
        avg_negative = sum(negative_values) / len(negative_values)
    else:
        avg_negative = 0.0
    
    # Calculate raw mood (-1 to +1 range)
    raw_mood = happy_confidence - avg_negative
    
    # Clamp raw_mood between -1 and +1
    raw_mood = max(-1.0, min(raw_mood, 1.0))
    
    # Scale to 0-100 range
    mood_score = int(round((raw_mood + 1) * 50))
    
    logger.debug(
        f"Mood score calculation: Happy={happy_confidence:.3f}, "
        f"Negatives=[Sad={sad_confidence:.3f}, Angry={anger_confidence:.3f}, Fear={fear_confidence:.3f}], "
        f"AvgNegative={avg_negative:.3f}, RawMood={raw_mood:.3f}, MoodScore={mood_score}"
    )
    
    return mood_score


def calculate_negative_emotion_consensus(
    modality_scores: Dict[str, Dict[str, float]],
    weights: Dict[str, float]
) -> Tuple[float, int]:
    """
    Calculate negative emotion consensus score and frequency.
    
    This detects when multiple modalities show negative emotions (even if different ones).
    The idea is that mixed negative emotions (Sad, Angry, Fear) across modalities
    indicate a critical emotional state that should boost confidence, not reduce it.
    
    Args:
        modality_scores: Dictionary mapping modality -> emotion -> avg_confidence
        weights: Dictionary mapping modality -> weight
    
    Returns:
        Tuple of (consensus_score: float, negative_modality_count: int)
        - consensus_score: Weighted sum of negative emotion confidences across all modalities (0.0-1.0)
        - negative_modality_count: Number of modalities showing negative emotions
    """
    negative_consensus = 0.0
    negative_modality_count = 0
    
    for modality, emotion_scores in modality_scores.items():
        modality_weight = weights.get(modality, 0.0)
        if modality_weight == 0.0:
            continue
        
        # Check if this modality has any negative emotions
        modality_negative_sum = 0.0
        for emotion, confidence in emotion_scores.items():
            if emotion in NEGATIVE_EMOTIONS:
                modality_negative_sum += confidence
        
        if modality_negative_sum > 0.0:
            negative_modality_count += 1
            # Weight the negative emotion contribution by modality weight
            negative_consensus += modality_negative_sum * modality_weight
    
    return negative_consensus, negative_modality_count


def apply_negative_emotion_boost(
    confidence_score: float,
    emotion_label: str,
    negative_consensus: float,
    negative_modality_count: int,
    total_modalities: int
) -> float:
    """
    Apply confidence boost when negative emotions are detected across multiple modalities.
    
    The boost increases confidence when:
    1. The selected emotion is negative
    2. Multiple modalities show negative emotions (consensus)
    3. The negative consensus score is high
    
    This ensures that mixed negative emotions (e.g., SER=Sad, FER=Angry, Vitals=Fear)
    are treated as a critical pattern that increases confidence, not decreases it.
    
    Args:
        confidence_score: Base confidence score (0.0-1.0)
        emotion_label: Selected emotion label
        negative_consensus: Weighted sum of negative emotion confidences
        negative_modality_count: Number of modalities showing negative emotions
        total_modalities: Total number of contributing modalities
    
    Returns:
        Boosted confidence score (0.0-1.0)
    """
    # Only boost if the selected emotion is negative
    if emotion_label not in NEGATIVE_EMOTIONS:
        return confidence_score
    
    # Calculate boost factor based on negative emotion frequency/consensus
    # More modalities showing negative emotions = higher boost
    if total_modalities == 0:
        return confidence_score
    
    # Frequency factor: how many modalities show negative emotions
    frequency_factor = negative_modality_count / total_modalities
    
    # Consensus factor: how strong the negative emotions are
    consensus_factor = min(negative_consensus, 1.0)
    
    # Boost multiplier: increases with both frequency and consensus
    # Formula: 1.0 + (frequency_factor * consensus_factor * boost_strength)
    # boost_strength controls how much we boost (0.0 = no boost, 1.0 = max boost)
    boost_multiplier = 1.0 + (frequency_factor * consensus_factor * NEGATIVE_EMOTION_BOOST_STRENGTH)
    
    # Apply boost
    boosted_confidence = min(confidence_score * boost_multiplier, 1.0)
    
    logger.info(
        f"Negative emotion boost applied: base={confidence_score:.3f}, "
        f"frequency={frequency_factor:.2f} ({negative_modality_count}/{total_modalities}), "
        f"consensus={consensus_factor:.3f}, multiplier={boost_multiplier:.3f}, "
        f"boosted={boosted_confidence:.3f}"
    )
    
    return boosted_confidence


def fuse_signals(signals: List[ModelSignal], weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Fuse multiple emotion signals using weighted aggregation.
    
    Algorithm:
    1. Group signals by modality
    2. For each modality, aggregate predictions (average confidence per emotion)
    3. Apply modality weights
    4. Calculate weighted score per emotion: sum(modality_avg_confidence * weight)
    5. Select emotion with highest score
    6. Normalize confidence_score to [0, 1]
    7. Calculate emotional_score [0, 100] using balanced mood score from all emotions
    
    Args:
        signals: List of ModelSignal objects from different modalities
        weights: Optional weights dictionary. If None, uses config weights.
    
    Returns:
        Dictionary with:
        - emotion_label: str
        - confidence_score: float (0.0 to 1.0)
        - emotional_score: int (0 to 100) - overall mood valence score (0=very negative, 50=neutral, 100=very positive)
        - signals_used: List[Dict] with modality, emotion_label, confidence
    """
    if not signals:
        raise ValueError("Cannot fuse empty signals list")
    
    weights = weights or FUSION_WEIGHTS
    
    # Step 1: Group signals by modality
    modality_signals = defaultdict(list)
    for signal in signals:
        modality_signals[signal.modality].append(signal)
    
    logger.debug(f"Grouped signals by modality: {dict((k, len(v)) for k, v in modality_signals.items())}")
    
    # Step 2: Aggregate predictions per modality (average confidence per emotion)
    modality_emotions = defaultdict(lambda: defaultdict(list))
    
    for modality, modality_signal_list in modality_signals.items():
        for signal in modality_signal_list:
            # Validate emotion label
            if signal.emotion_label not in VALID_EMOTIONS:
                logger.warning(f"Invalid emotion label '{signal.emotion_label}' from {modality}, skipping")
                continue
            
            modality_emotions[modality][signal.emotion_label].append(signal.confidence)
    
    # Step 3: Calculate average confidence per emotion per modality
    modality_scores = defaultdict(dict)
    
    for modality, emotions_dict in modality_emotions.items():
        for emotion, confidences in emotions_dict.items():
            avg_confidence = sum(confidences) / len(confidences)
            modality_scores[modality][emotion] = avg_confidence
            logger.debug(f"{modality} -> {emotion}: avg_confidence={avg_confidence:.3f} (from {len(confidences)} signals)")
    
    # Step 4: Apply weights and calculate weighted scores per emotion
    emotion_weighted_scores = defaultdict(float)
    
    for modality, emotion_scores in modality_scores.items():
        modality_weight = weights.get(modality, 0.0)
        if modality_weight == 0.0:
            logger.warning(f"Modality '{modality}' has zero weight, skipping")
            continue
        
        for emotion, avg_confidence in emotion_scores.items():
            weighted_contribution = avg_confidence * modality_weight
            emotion_weighted_scores[emotion] += weighted_contribution
            logger.debug(f"{emotion} += {modality} ({avg_confidence:.3f} * {modality_weight:.2f}) = {weighted_contribution:.3f}")
    
    if not emotion_weighted_scores:
        raise ValueError("No valid emotion scores after aggregation")
    
    # Step 5: Select emotion with highest score
    best_emotion = max(emotion_weighted_scores.items(), key=lambda x: x[1])
    emotion_label = best_emotion[0]
    raw_score = best_emotion[1]
    
    logger.info(f"Selected emotion '{emotion_label}' with raw weighted score {raw_score:.3f}")
    
    # Step 6: Normalize confidence_score to [0, 1]
    # The raw score is already weighted, but we need to normalize it
    # Since weights sum to 1.0, and we're averaging confidences, the max possible score is 1.0
    # However, if not all modalities contribute, the max might be less
    # We'll normalize by dividing by the sum of weights that contributed
    contributing_weights_sum = sum(
        weights.get(modality, 0.0)
        for modality in modality_scores.keys()
    )
    
    if contributing_weights_sum > 0:
        confidence_score = min(raw_score / contributing_weights_sum, 1.0)
    else:
        confidence_score = 0.0
    
    # Step 6.5: Apply negative emotion boost if applicable
    # This boosts confidence when multiple modalities detect negative emotions
    # (even if different negative emotions), treating it as a critical pattern
    if NEGATIVE_EMOTION_BOOST_ENABLED:
        negative_consensus, negative_modality_count = calculate_negative_emotion_consensus(
            modality_scores, weights
        )
        total_modalities = len(modality_scores)
        
        confidence_score = apply_negative_emotion_boost(
            confidence_score,
            emotion_label,
            negative_consensus,
            negative_modality_count,
            total_modalities
        )
    
    # Step 7: Calculate emotional_score [0, 100] using balanced mood score
    # Normalize all emotion weighted scores to get confidences for mood calculation
    normalized_emotion_confidences = {}
    for emotion, raw_weighted_score in emotion_weighted_scores.items():
        normalized_emotion_confidences[emotion] = min(raw_weighted_score / contributing_weights_sum, 1.0)
    
    # Initialize missing emotions to 0.0
    for emotion in VALID_EMOTIONS:
        if emotion not in normalized_emotion_confidences:
            normalized_emotion_confidences[emotion] = 0.0
    
    # Calculate mood score from all normalized emotion confidences
    emotional_score = calculate_mood_score(normalized_emotion_confidences)
    
    # Build signals_used list for response
    signals_used = []
    for modality, emotion_scores in modality_scores.items():
        for emotion, avg_confidence in emotion_scores.items():
            signals_used.append({
                "modality": modality,
                "emotion_label": emotion,
                "confidence": round(avg_confidence, 3)
            })
    
    result = {
        "emotion_label": emotion_label,
        "confidence_score": round(confidence_score, 3),
        "emotional_score": emotional_score,
        "signals_used": signals_used
    }
    
    logger.info(f"Fused result: {emotion_label} (confidence: {confidence_score:.3f}, emotional_score: {emotional_score}) - mood valence score")
    
    return result


def map_fused_to_db_row(fused: Dict) -> Dict:
    """
    Map fused result to database row format.
    
    Args:
        fused: Dictionary from fuse_signals() output
    
    Returns:
        Dictionary with database-compatible fields
    """
    return {
        "emotion_label": fused["emotion_label"],
        "confidence_score": fused["confidence_score"],
        "emotional_score": fused["emotional_score"]
    }

