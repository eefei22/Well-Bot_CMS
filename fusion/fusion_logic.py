"""
Core Fusion Logic for Fusion Service

This module implements the weighted fusion algorithm that aggregates
emotion predictions from multiple modalities.
"""

import logging
from typing import List, Dict, Optional
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

# Valid emotion labels
VALID_EMOTIONS = ["Angry", "Sad", "Happy", "Fear"]


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
    7. Map to emotional_score [0, 100]
    
    Args:
        signals: List of ModelSignal objects from different modalities
        weights: Optional weights dictionary. If None, uses config weights.
    
    Returns:
        Dictionary with:
        - emotion_label: str
        - confidence_score: float (0.0 to 1.0)
        - emotional_score: int (0 to 100)
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
    
    # Step 7: Map to emotional_score [0, 100]
    emotional_score = int(round(confidence_score * 100))
    
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
    
    logger.info(f"Fused result: {emotion_label} (confidence: {confidence_score:.3f}, emotional_score: {emotional_score})")
    
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

