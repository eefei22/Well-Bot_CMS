"""
Unit Tests: fuse_signals() Conflict Resolution Scenarios

Tests the weighted fusion algorithm's behavior when modalities disagree.

Weights: speech=0.4, face=0.3, vitals=0.3
Formula: weighted_score = confidence × weight
Winner: emotion with highest total weighted score

Run with: pytest test_fuse_signals_conflicts.py -v
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.models import ModelSignal
from fusion.fusion_logic import fuse_signals


# =============================================================================
# Test Fixtures & Helpers
# =============================================================================

def make_signal(modality: str, emotion: str, confidence: float) -> ModelSignal:
    """Helper to create a ModelSignal with minimal boilerplate."""
    return ModelSignal(
        user_id="test-user-00000000-0000-0000-0000-000000000000",
        timestamp="2026-01-13T10:00:00+08:00",
        modality=modality,
        emotion_label=emotion,
        confidence=confidence
    )


# Standard weights for manual calculations
WEIGHTS = {"speech": 0.4, "face": 0.3, "vitals": 0.3}


# =============================================================================
# Test Suite: Conflict Resolution Scenarios
# =============================================================================

class TestConflictResolution:
    """
    Tests for fuse_signals() when modalities report different emotions.
    
    These tests verify the weighted aggregation algorithm correctly
    resolves conflicts based on confidence × weight scoring.
    """

    # -------------------------------------------------------------------------
    # Scenario 1: All Modalities Agree
    # -------------------------------------------------------------------------
    def test_all_modalities_agree_happy(self):
        """
        When all modalities agree on the same emotion, that emotion wins.
        
        Input:
            Speech: Happy@0.8  → 0.8 × 0.4 = 0.32
            Face:   Happy@0.8  → 0.8 × 0.3 = 0.24
            Vitals: Happy@0.8  → 0.8 × 0.3 = 0.24
        
        Expected: Happy wins (total = 0.80)
        """
        signals = [
            make_signal("speech", "Happy", 0.8),
            make_signal("face", "Happy", 0.8),
            make_signal("vitals", "Happy", 0.8),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"
        assert result["confidence_score"] == pytest.approx(0.8, abs=0.01)

    def test_all_modalities_agree_sad(self):
        """
        Agreement case with negative emotion (Sad).
        
        Input:
            Speech: Sad@0.7  → 0.7 × 0.4 = 0.28
            Face:   Sad@0.7  → 0.7 × 0.3 = 0.21
            Vitals: Sad@0.7  → 0.7 × 0.3 = 0.21
        
        Expected: Sad wins (total = 0.70)
        """
        signals = [
            make_signal("speech", "Sad", 0.7),
            make_signal("face", "Sad", 0.7),
            make_signal("vitals", "Sad", 0.7),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Sad"
        assert result["confidence_score"] == pytest.approx(0.7, abs=0.01)

    # -------------------------------------------------------------------------
    # Scenario 2: Two vs One Disagreement
    # -------------------------------------------------------------------------
    def test_two_vs_one_face_vitals_win(self):
        """
        Two modalities (face + vitals) agree against speech.
        Combined weight 0.6 beats speech's 0.4.
        
        Input:
            Speech: Happy@0.9  → 0.9 × 0.4 = 0.36
            Face:   Sad@0.9    → 0.9 × 0.3 = 0.27
            Vitals: Sad@0.9    → 0.9 × 0.3 = 0.27
        
        Expected: Sad wins (0.54 > 0.36)
        """
        signals = [
            make_signal("speech", "Happy", 0.9),
            make_signal("face", "Sad", 0.9),
            make_signal("vitals", "Sad", 0.9),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Sad"
        # Confidence normalized by contributing weights (1.0)
        # Sad raw = 0.54, normalized = 0.54/1.0 = 0.54
        assert result["confidence_score"] == pytest.approx(0.54, abs=0.01)

    def test_two_vs_one_speech_face_win(self):
        """
        Speech + Face agree against Vitals.
        Combined weight 0.7 beats vitals' 0.3.
        
        Input:
            Speech: Angry@0.8  → 0.8 × 0.4 = 0.32
            Face:   Angry@0.8  → 0.8 × 0.3 = 0.24
            Vitals: Fear@0.9   → 0.9 × 0.3 = 0.27
        
        Expected: Angry wins (0.56 > 0.27)
        """
        signals = [
            make_signal("speech", "Angry", 0.8),
            make_signal("face", "Angry", 0.8),
            make_signal("vitals", "Fear", 0.9),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Angry"
        assert result["confidence_score"] == pytest.approx(0.56, abs=0.01)

    # -------------------------------------------------------------------------
    # Scenario 3: Weight Dominance (Speech's Higher Weight)
    # -------------------------------------------------------------------------
    def test_speech_dominance_high_confidence(self):
        """
        Speech (weight 0.4) can beat face+vitals (0.6) with higher confidence.
        
        Input:
            Speech: Happy@1.0  → 1.0 × 0.4 = 0.40
            Face:   Angry@0.6  → 0.6 × 0.3 = 0.18
            Vitals: Angry@0.6  → 0.6 × 0.3 = 0.18
        
        Expected: Happy wins (0.40 > 0.36)
        """
        signals = [
            make_signal("speech", "Happy", 1.0),
            make_signal("face", "Angry", 0.6),
            make_signal("vitals", "Angry", 0.6),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"
        assert result["confidence_score"] == pytest.approx(0.40, abs=0.01)

    def test_speech_dominance_threshold(self):
        """
        Find the threshold where speech alone beats face+vitals combined.
        
        For speech to win: speech_conf × 0.4 > other_conf × 0.6
        If other_conf = 0.9: speech needs > 0.9 × 0.6 / 0.4 = 1.35 (impossible)
        If other_conf = 0.6: speech needs > 0.6 × 0.6 / 0.4 = 0.9
        
        Input:
            Speech: Happy@0.95 → 0.95 × 0.4 = 0.38
            Face:   Sad@0.6    → 0.6 × 0.3 = 0.18
            Vitals: Sad@0.6    → 0.6 × 0.3 = 0.18
        
        Expected: Happy wins (0.38 > 0.36)
        """
        signals = [
            make_signal("speech", "Happy", 0.95),
            make_signal("face", "Sad", 0.6),
            make_signal("vitals", "Sad", 0.6),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"

    def test_speech_dominance_fails_at_equal_confidence(self):
        """
        At equal confidence, face+vitals (0.6 weight) beats speech (0.4 weight).
        
        Input:
            Speech: Happy@0.8  → 0.8 × 0.4 = 0.32
            Face:   Sad@0.8    → 0.8 × 0.3 = 0.24
            Vitals: Sad@0.8    → 0.8 × 0.3 = 0.24
        
        Expected: Sad wins (0.48 > 0.32)
        """
        signals = [
            make_signal("speech", "Happy", 0.8),
            make_signal("face", "Sad", 0.8),
            make_signal("vitals", "Sad", 0.8),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Sad"

    # -------------------------------------------------------------------------
    # Scenario 4: All Three Modalities Disagree
    # -------------------------------------------------------------------------
    def test_all_disagree_speech_wins(self):
        """
        All modalities report different emotions.
        Speech wins due to higher weight at similar confidence.
        
        Input:
            Speech: Happy@0.8  → 0.8 × 0.4 = 0.32
            Face:   Sad@0.7    → 0.7 × 0.3 = 0.21
            Vitals: Angry@0.6  → 0.6 × 0.3 = 0.18
        
        Expected: Happy wins (0.32 is highest)
        """
        signals = [
            make_signal("speech", "Happy", 0.8),
            make_signal("face", "Sad", 0.7),
            make_signal("vitals", "Angry", 0.6),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"
        assert result["confidence_score"] == pytest.approx(0.32, abs=0.01)

    def test_all_disagree_face_wins_high_confidence(self):
        """
        Face wins despite lower weight due to much higher confidence.
        
        Input:
            Speech: Happy@0.5  → 0.5 × 0.4 = 0.20
            Face:   Sad@0.95   → 0.95 × 0.3 = 0.285
            Vitals: Angry@0.6  → 0.6 × 0.3 = 0.18
        
        Expected: Sad wins (0.285 is highest)
        """
        signals = [
            make_signal("speech", "Happy", 0.5),
            make_signal("face", "Sad", 0.95),
            make_signal("vitals", "Angry", 0.6),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Sad"

    def test_all_disagree_four_emotions(self):
        """
        Each modality reports one of the four valid emotions.
        (Vitals sends two signals with different emotions)
        
        Input:
            Speech: Happy@0.7  → 0.7 × 0.4 = 0.28
            Face:   Sad@0.8    → 0.8 × 0.3 = 0.24
            Vitals: Angry@0.5  → 0.5 × 0.3 = 0.15
            Vitals: Fear@0.3   → 0.3 × 0.3 = 0.09
        
        Note: Vitals averages Angry and Fear separately.
        Expected: Happy wins (0.28 is highest single emotion)
        """
        signals = [
            make_signal("speech", "Happy", 0.7),
            make_signal("face", "Sad", 0.8),
            make_signal("vitals", "Angry", 0.5),
            make_signal("vitals", "Fear", 0.3),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"

    # -------------------------------------------------------------------------
    # Scenario 5: Close Contest / Near-Tie
    # -------------------------------------------------------------------------
    def test_close_contest_speech_wins_narrowly(self):
        """
        Very close weighted scores - speech wins by tiny margin.
        
        Input:
            Speech: Happy@0.8  → 0.8 × 0.4 = 0.320
            Face:   Sad@1.0    → 1.0 × 0.3 = 0.300
            Vitals: Angry@0.05 → 0.05 × 0.3 = 0.015
        
        Expected: Happy wins (0.32 > 0.30)
        """
        signals = [
            make_signal("speech", "Happy", 0.8),
            make_signal("face", "Sad", 1.0),
            make_signal("vitals", "Angry", 0.05),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"

    def test_close_contest_face_wins_narrowly(self):
        """
        Face wins by tiny margin over speech.
        
        Input:
            Speech: Happy@0.74 → 0.74 × 0.4 = 0.296
            Face:   Sad@1.0    → 1.0 × 0.3 = 0.300
            Vitals: Angry@0.05 → 0.05 × 0.3 = 0.015
        
        Expected: Sad wins (0.30 > 0.296)
        """
        signals = [
            make_signal("speech", "Happy", 0.74),
            make_signal("face", "Sad", 1.0),
            make_signal("vitals", "Angry", 0.05),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Sad"

    def test_exact_tie_deterministic(self):
        """
        When two emotions have exactly equal scores, result should be deterministic.
        (Python's max() returns first item found in case of tie)
        
        Input:
            Speech: Happy@0.75 → 0.75 × 0.4 = 0.30
            Face:   Sad@1.0    → 1.0 × 0.3 = 0.30
        
        Expected: Deterministic result (either Happy or Sad, but consistent)
        """
        signals = [
            make_signal("speech", "Happy", 0.75),
            make_signal("face", "Sad", 1.0),
        ]
        
        # Run multiple times to verify determinism
        results = [fuse_signals(signals, WEIGHTS)["emotion_label"] for _ in range(5)]
        
        # All results should be identical
        assert len(set(results)) == 1, f"Non-deterministic results: {results}"


# =============================================================================
# Test Suite: Partial Modality Scenarios (Conflict-Adjacent)
# =============================================================================

class TestPartialModalityConflicts:
    """
    Tests where only some modalities are present, affecting conflict dynamics.
    """

    def test_only_speech_present(self):
        """
        Only speech modality - no conflict possible, speech wins by default.
        
        Input:
            Speech: Angry@0.85 → 0.85 × 0.4 = 0.34
        
        Expected: Angry wins, confidence normalized by contributing weight (0.4)
        Normalized confidence = 0.34 / 0.4 = 0.85
        """
        signals = [
            make_signal("speech", "Angry", 0.85),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Angry"
        assert result["confidence_score"] == pytest.approx(0.85, abs=0.01)

    def test_speech_vs_face_only(self):
        """
        Only speech and face present - conflict with partial weights.
        
        Input:
            Speech: Happy@0.8  → 0.8 × 0.4 = 0.32
            Face:   Sad@0.9    → 0.9 × 0.3 = 0.27
        
        Contributing weights = 0.4 + 0.3 = 0.7
        Expected: Happy wins (0.32 > 0.27)
        Normalized confidence = 0.32 / 0.7 ≈ 0.457
        """
        signals = [
            make_signal("speech", "Happy", 0.8),
            make_signal("face", "Sad", 0.9),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Happy"
        assert result["confidence_score"] == pytest.approx(0.457, abs=0.01)

    def test_face_vs_vitals_only(self):
        """
        Only face and vitals present (no speech) - equal weights conflict.
        
        Input:
            Face:   Happy@0.7  → 0.7 × 0.3 = 0.21
            Vitals: Sad@0.8    → 0.8 × 0.3 = 0.24
        
        Contributing weights = 0.3 + 0.3 = 0.6
        Expected: Sad wins (0.24 > 0.21)
        Normalized confidence = 0.24 / 0.6 = 0.40
        """
        signals = [
            make_signal("face", "Happy", 0.7),
            make_signal("vitals", "Sad", 0.8),
        ]
        
        result = fuse_signals(signals, WEIGHTS)
        
        assert result["emotion_label"] == "Sad"
        assert result["confidence_score"] == pytest.approx(0.40, abs=0.01)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
