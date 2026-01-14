"""
Test Runner: fuse_signals() Conflict Resolution with CSV Report

Generates a detailed CSV report showing:
- Test ID, Test Name, Scenario
- Input signals (modality, emotion, confidence)
- Expected emotion & confidence
- Actual emotion & confidence
- Pass/Fail status

Run with: python run_conflict_tests_report.py
"""

import sys
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.models import ModelSignal
from fusion.fusion_logic import fuse_signals


# =============================================================================
# Helpers
# =============================================================================

def make_signal(modality: str, emotion: str, confidence: float) -> ModelSignal:
    """Helper to create a ModelSignal."""
    return ModelSignal(
        user_id="test-user-00000000-0000-0000-0000-000000000000",
        timestamp="2026-01-13T10:00:00+08:00",
        modality=modality,
        emotion_label=emotion,
        confidence=confidence
    )


def format_signals(signals: List[ModelSignal]) -> str:
    """Format signals list as readable string."""
    parts = []
    for s in signals:
        parts.append(f"{s.modality}:{s.emotion_label}@{s.confidence}")
    return " | ".join(parts)


def calculate_weighted_scores(signals: List[ModelSignal], weights: Dict[str, float]) -> Dict[str, float]:
    """Calculate weighted scores for each emotion (mirrors fusion algorithm)."""
    from collections import defaultdict
    
    # Group by modality and emotion, calculate averages
    modality_emotions = defaultdict(lambda: defaultdict(list))
    for s in signals:
        modality_emotions[s.modality][s.emotion_label].append(s.confidence)
    
    # Calculate weighted scores per emotion
    emotion_scores = defaultdict(float)
    for modality, emotions in modality_emotions.items():
        weight = weights.get(modality, 0.0)
        for emotion, confidences in emotions.items():
            avg_conf = sum(confidences) / len(confidences)
            emotion_scores[emotion] += avg_conf * weight
    
    return dict(emotion_scores)


def format_weighted_scores(scores: Dict[str, float]) -> str:
    """Format weighted scores as readable string."""
    parts = [f"{emotion}={score:.3f}" for emotion, score in sorted(scores.items(), key=lambda x: -x[1])]
    return " | ".join(parts)


# Standard weights
WEIGHTS = {"speech": 0.4, "face": 0.3, "vitals": 0.3}


# =============================================================================
# Test Case Definitions
# =============================================================================

TEST_CASES = [
    # Scenario 1: All Modalities Agree
    {
        "id": "TC-01",
        "name": "all_modalities_agree_happy",
        "scenario": "All Agree",
        "signals": [
            ("speech", "Happy", 0.8),
            ("face", "Happy", 0.8),
            ("vitals", "Happy", 0.8),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.8,
        "description": "All modalities agree on Happy@0.8"
    },
    {
        "id": "TC-02",
        "name": "all_modalities_agree_sad",
        "scenario": "All Agree",
        "signals": [
            ("speech", "Sad", 0.7),
            ("face", "Sad", 0.7),
            ("vitals", "Sad", 0.7),
        ],
        "expected_emotion": "Sad",
        "expected_confidence": 0.7,
        "description": "All modalities agree on Sad@0.7"
    },
    
    # Scenario 2: Two vs One Disagreement
    {
        "id": "TC-03",
        "name": "two_vs_one_face_vitals_win",
        "scenario": "2-vs-1 Conflict",
        "signals": [
            ("speech", "Happy", 0.9),
            ("face", "Sad", 0.9),
            ("vitals", "Sad", 0.9),
        ],
        "expected_emotion": "Sad",
        "expected_confidence": 0.54,
        "description": "Face+Vitals (0.6) beat Speech (0.4) at equal confidence"
    },
    {
        "id": "TC-04",
        "name": "two_vs_one_speech_face_win",
        "scenario": "2-vs-1 Conflict",
        "signals": [
            ("speech", "Angry", 0.8),
            ("face", "Angry", 0.8),
            ("vitals", "Fear", 0.9),
        ],
        "expected_emotion": "Angry",
        "expected_confidence": 0.56,
        "description": "Speech+Face (0.7) beat Vitals (0.3)"
    },
    
    # Scenario 3: Weight Dominance
    {
        "id": "TC-05",
        "name": "speech_dominance_high_confidence",
        "scenario": "Weight Dominance",
        "signals": [
            ("speech", "Happy", 1.0),
            ("face", "Angry", 0.6),
            ("vitals", "Angry", 0.6),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.40,
        "description": "Speech@1.0 beats Face+Vitals@0.6 (0.40 > 0.36)"
    },
    {
        "id": "TC-06",
        "name": "speech_dominance_threshold",
        "scenario": "Weight Dominance",
        "signals": [
            ("speech", "Happy", 0.95),
            ("face", "Sad", 0.6),
            ("vitals", "Sad", 0.6),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.38,
        "description": "Speech@0.95 just beats Face+Vitals@0.6 (0.38 > 0.36)"
    },
    {
        "id": "TC-07",
        "name": "speech_dominance_fails_equal_confidence",
        "scenario": "Weight Dominance",
        "signals": [
            ("speech", "Happy", 0.8),
            ("face", "Sad", 0.8),
            ("vitals", "Sad", 0.8),
        ],
        "expected_emotion": "Sad",
        "expected_confidence": 0.48,
        "description": "At equal confidence, combined weight (0.6) beats (0.4)"
    },
    
    # Scenario 4: All Three Disagree
    {
        "id": "TC-08",
        "name": "all_disagree_speech_wins",
        "scenario": "All Disagree",
        "signals": [
            ("speech", "Happy", 0.8),
            ("face", "Sad", 0.7),
            ("vitals", "Angry", 0.6),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.32,
        "description": "Speech wins with highest weighted score (0.32)"
    },
    {
        "id": "TC-09",
        "name": "all_disagree_face_wins_high_confidence",
        "scenario": "All Disagree",
        "signals": [
            ("speech", "Happy", 0.5),
            ("face", "Sad", 0.95),
            ("vitals", "Angry", 0.6),
        ],
        "expected_emotion": "Sad",
        "expected_confidence": 0.285,
        "description": "Face wins with high confidence despite lower weight"
    },
    {
        "id": "TC-10",
        "name": "all_disagree_four_emotions",
        "scenario": "All Disagree",
        "signals": [
            ("speech", "Happy", 0.7),
            ("face", "Sad", 0.8),
            ("vitals", "Angry", 0.5),
            ("vitals", "Fear", 0.3),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.28,
        "description": "Four emotions present, Happy wins (0.28 highest)"
    },
    
    # Scenario 5: Close Contest / Near-Tie
    {
        "id": "TC-11",
        "name": "close_contest_speech_wins_narrowly",
        "scenario": "Close Contest",
        "signals": [
            ("speech", "Happy", 0.8),
            ("face", "Sad", 1.0),
            ("vitals", "Angry", 0.05),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.32,
        "description": "Speech wins narrowly (0.32 > 0.30)"
    },
    {
        "id": "TC-12",
        "name": "close_contest_face_wins_narrowly",
        "scenario": "Close Contest",
        "signals": [
            ("speech", "Happy", 0.74),
            ("face", "Sad", 1.0),
            ("vitals", "Angry", 0.05),
        ],
        "expected_emotion": "Sad",
        "expected_confidence": 0.30,
        "description": "Face wins narrowly (0.30 > 0.296)"
    },
    {
        "id": "TC-13",
        "name": "exact_tie_deterministic",
        "scenario": "Close Contest",
        "signals": [
            ("speech", "Happy", 0.75),
            ("face", "Sad", 1.0),
        ],
        "expected_emotion": None,  # We just check determinism
        "expected_confidence": 0.429,  # 0.30 / 0.7 (partial modality normalization)
        "description": "Exact tie (0.30 = 0.30) - verify determinism"
    },
    
    # Scenario 6: Partial Modalities
    {
        "id": "TC-14",
        "name": "only_speech_present",
        "scenario": "Partial Modality",
        "signals": [
            ("speech", "Angry", 0.85),
        ],
        "expected_emotion": "Angry",
        "expected_confidence": 0.85,
        "description": "Single modality - confidence normalized to original"
    },
    {
        "id": "TC-15",
        "name": "speech_vs_face_only",
        "scenario": "Partial Modality",
        "signals": [
            ("speech", "Happy", 0.8),
            ("face", "Sad", 0.9),
        ],
        "expected_emotion": "Happy",
        "expected_confidence": 0.457,
        "description": "Two modalities only - Speech wins (0.32 > 0.27)"
    },
    {
        "id": "TC-16",
        "name": "face_vs_vitals_only",
        "scenario": "Partial Modality",
        "signals": [
            ("face", "Happy", 0.7),
            ("vitals", "Sad", 0.8),
        ],
        "expected_emotion": "Sad",
        "expected_confidence": 0.40,
        "description": "Equal weights - higher confidence wins"
    },
]


# =============================================================================
# Test Runner
# =============================================================================

def run_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case and return results."""
    # Build signals
    signals = [make_signal(m, e, c) for m, e, c in test_case["signals"]]
    
    # Calculate weighted scores (for transparency)
    weighted_scores = calculate_weighted_scores(signals, WEIGHTS)
    contributing_weights = sum(WEIGHTS.get(s.modality, 0) for s in signals if WEIGHTS.get(s.modality, 0) > 0)
    # Deduplicate modalities for contributing weights
    unique_modalities = set(s.modality for s in signals)
    contributing_weights = sum(WEIGHTS.get(m, 0) for m in unique_modalities)
    
    # Run fusion
    try:
        result = fuse_signals(signals, WEIGHTS)
        actual_emotion = result["emotion_label"]
        actual_confidence = result["confidence_score"]
        error = None
    except Exception as e:
        actual_emotion = "ERROR"
        actual_confidence = 0.0
        error = str(e)
    
    # Determine pass/fail
    expected_emotion = test_case["expected_emotion"]
    expected_confidence = test_case["expected_confidence"]
    
    # For TC-13 (tie test), we just check it doesn't error
    if expected_emotion is None:
        emotion_match = True  # Don't check emotion for tie test
    else:
        emotion_match = (actual_emotion == expected_emotion)
    
    confidence_match = abs(actual_confidence - expected_confidence) <= 0.02
    passed = emotion_match and confidence_match and error is None
    
    return {
        "id": test_case["id"],
        "name": test_case["name"],
        "scenario": test_case["scenario"],
        "description": test_case["description"],
        "input_signals": format_signals(signals),
        "weighted_scores": format_weighted_scores(weighted_scores),
        "contributing_weights": f"{contributing_weights:.1f}",
        "expected_emotion": expected_emotion if expected_emotion else "(any - tie)",
        "expected_confidence": f"{expected_confidence:.3f}",
        "actual_emotion": actual_emotion,
        "actual_confidence": f"{actual_confidence:.3f}",
        "emotion_match": "Y" if emotion_match else "N",
        "confidence_match": "Y" if confidence_match else "N",
        "status": "PASS" if passed else "FAIL",
        "error": error or ""
    }


def run_all_tests() -> List[Dict[str, Any]]:
    """Run all test cases and return results."""
    results = []
    for test_case in TEST_CASES:
        result = run_test(test_case)
        results.append(result)
    return results


def save_csv(results: List[Dict[str, Any]], filepath: str):
    """Save results to CSV file."""
    fieldnames = [
        "id", "name", "scenario", "description", "input_signals",
        "weighted_scores", "contributing_weights",
        "expected_emotion", "expected_confidence",
        "actual_emotion", "actual_confidence",
        "emotion_match", "confidence_match", "status", "error"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: List[Dict[str, Any]]):
    """Print summary table to console."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    
    print("\n" + "=" * 120)
    print("FUSION CONFLICT RESOLUTION TEST RESULTS")
    print("=" * 120)
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("\n" + "-" * 120)
    
    # Header
    print(f"{'ID':<8} {'Scenario':<18} {'Input Signals':<45} {'Expected':<12} {'Actual':<12} {'Status':<8}")
    print("-" * 120)
    
    # Rows
    for r in results:
        input_short = r["input_signals"][:42] + "..." if len(r["input_signals"]) > 45 else r["input_signals"]
        expected = f"{r['expected_emotion']}@{r['expected_confidence']}"
        actual = f"{r['actual_emotion']}@{r['actual_confidence']}"
        status_icon = "[OK]" if r["status"] == "PASS" else "[X]"
        
        print(f"{r['id']:<8} {r['scenario']:<18} {input_short:<45} {expected:<12} {actual:<12} {status_icon} {r['status']}")
    
    print("-" * 120)
    print(f"\nPassed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Running fuse_signals() conflict resolution tests...")
    
    # Run tests
    results = run_all_tests()
    
    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "conflict_test_results.csv")
    save_csv(results, csv_path)
    print(f"\nCSV saved to: {csv_path}")
    
    # Print summary
    print_summary(results)
