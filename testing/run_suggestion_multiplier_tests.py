"""
Test Runner: suggest_activities() Multiplier Stacking Scenarios

Tests how emotion and frequency multipliers affect activity rankings.

Focus:
    - Group 1: Emotion variation (preferences & counts constant)
    - Group 2: Frequency variation (emotion & preferences constant)

Constants:
    - Preferences: All True (uniform 1.2x boost for all activities)
    - This isolates emotion and frequency effects

Algorithm:
    final_score = base_weight * preference_multiplier * frequency_multiplier
    (then normalized to [0, 1])

Multipliers:
    - Preference: preferred=1.2x, not_preferred=0.7x
    - Frequency: rank1=1.3x, rank2=1.2x, rank3=1.1x, rank4=1.05x

Run with: python run_suggestion_multiplier_tests.py
"""

import sys
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervention.suggestion_engine import suggest_activities

# =============================================================================
# Constants (matching config)
# =============================================================================

EMOTION_WEIGHTS = {
    'Sad': {'journal': 0.9, 'meditation': 0.8, 'gratitude': 0.7, 'quote': 0.6},
    'Angry': {'meditation': 0.9, 'journal': 0.7, 'quote': 0.6, 'gratitude': 0.5},
    'Fear': {'meditation': 0.8, 'quote': 0.7, 'journal': 0.7, 'gratitude': 0.6},
    'Happy': {'gratitude': 0.8, 'journal': 0.7, 'quote': 0.6, 'meditation': 0.5}
}

PREF_BOOST = 1.2
PREF_PENALTY = 0.7
FREQ_MULTIPLIERS = {1: 1.3, 2: 1.2, 3: 1.1, 4: 1.05}


# =============================================================================
# Helpers
# =============================================================================

def calc_raw_scores(emotion: str, preferences: Dict, activity_counts: Dict) -> Dict[str, float]:
    """
    Calculate raw scores (before normalization) for verification.
    Mirrors the algorithm logic.
    """
    base_weights = EMOTION_WEIGHTS.get(emotion, {'journal': 0.7, 'gratitude': 0.7, 'meditation': 0.7, 'quote': 0.7})
    
    # Start with base weights
    scores = dict(base_weights)
    
    # Apply preference multipliers
    pref_mapping = {'journaling': 'journal', 'gratitude': 'gratitude', 'breathing': 'meditation', 'quote': 'quote'}
    for pref_key, pref_value in preferences.items():
        activity = pref_mapping.get(pref_key)
        if activity and activity in scores:
            if pref_value:
                scores[activity] *= PREF_BOOST
            else:
                scores[activity] *= PREF_PENALTY
    
    # Apply frequency multipliers (group by count)
    counts = {a: activity_counts.get(a, 0) for a in scores.keys()}
    frequency_groups = {}
    for activity, count in counts.items():
        if count not in frequency_groups:
            frequency_groups[count] = []
        frequency_groups[count].append(activity)
    
    sorted_counts = sorted(frequency_groups.keys(), reverse=True)
    for group_rank, group_count in enumerate(sorted_counts, start=1):
        multiplier = FREQ_MULTIPLIERS.get(group_rank, 1.0)
        for activity in frequency_groups[group_count]:
            scores[activity] *= multiplier
    
    return scores


def format_preferences(prefs: Dict) -> str:
    """Format preferences as readable string."""
    if not prefs:
        return "(none)"
    parts = []
    for k, v in prefs.items():
        parts.append(f"{k}={'Y' if v else 'N'}")
    return ", ".join(parts)


def format_counts(counts: Dict) -> str:
    """Format activity counts as readable string."""
    if not counts:
        return "(all zero)"
    parts = [f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return ", ".join(parts)


def format_raw_scores(scores: Dict) -> str:
    """Format raw scores as readable string (sorted by value)."""
    parts = [f"{k}={v:.3f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])]
    return " | ".join(parts)


def format_ranking(ranked: List[Dict]) -> str:
    """Format ranked activities as readable string."""
    parts = [f"{r['rank']}:{r['activity_type']}({r['score']:.3f})" for r in ranked]
    return " > ".join(parts)


# =============================================================================
# Test Case Definitions
# =============================================================================

# Constants for all tests
ALL_PREFERENCES_TRUE = {"journaling": True, "breathing": True, "gratitude": True, "quote": True}
EQUAL_COUNTS = {"journal": 10, "meditation": 10, "gratitude": 10, "quote": 10}

TEST_CASES = [
    # =========================================================================
    # Group 1: Emotion Variation (Preferences & Activity Counts Constant)
    # =========================================================================
    # Purpose: Observe how different emotions produce different activity rankings
    # Fixed: preferences=all True, activity_counts=all equal (10 each)
    # Since all get same preference (1.2x) and frequency (1.3x), base weights determine ranking
    # -------------------------------------------------------------------------
    {
        "id": "EV-01",
        "name": "emotion_sad",
        "scenario": "Emotion Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": EQUAL_COUNTS,
        "expected_top": "journal",
        "expected_ranking": ["journal", "meditation", "gratitude", "quote"],
        "description": "Sad: journal(0.9) > meditation(0.8) > gratitude(0.7) > quote(0.6) - all get 1.2x pref, 1.3x freq"
    },
    {
        "id": "EV-02",
        "name": "emotion_angry",
        "scenario": "Emotion Variation",
        "emotion": "Angry",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": EQUAL_COUNTS,
        "expected_top": "meditation",
        "expected_ranking": ["meditation", "journal", "quote", "gratitude"],
        "description": "Angry: meditation(0.9) > journal(0.7) > quote(0.6) > gratitude(0.5) - all get 1.2x pref, 1.3x freq"
    },
    {
        "id": "EV-03",
        "name": "emotion_fear",
        "scenario": "Emotion Variation",
        "emotion": "Fear",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": EQUAL_COUNTS,
        "expected_top": "meditation",
        "expected_ranking": ["meditation", "journal", "quote", "gratitude"],  # journal and quote tied, journal sorts first
        "description": "Fear: meditation(0.8) > journal(0.7) = quote(0.7) > gratitude(0.6) - all get 1.2x pref, 1.3x freq"
    },
    {
        "id": "EV-04",
        "name": "emotion_happy",
        "scenario": "Emotion Variation",
        "emotion": "Happy",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": EQUAL_COUNTS,
        "expected_top": "gratitude",
        "expected_ranking": ["gratitude", "journal", "quote", "meditation"],
        "description": "Happy: gratitude(0.8) > journal(0.7) > quote(0.6) > meditation(0.5) - all get 1.2x pref, 1.3x freq"
    },
    
    # =========================================================================
    # Group 2: Frequency Variation (Emotion & Preferences Constant)
    # =========================================================================
    # Purpose: Observe how varying activity counts affects rankings
    # Fixed: emotion="Sad" (base: journal=0.9, meditation=0.8, gratitude=0.7, quote=0.6)
    # Fixed: preferences=all True (uniform 1.2x)
    # -------------------------------------------------------------------------
    {
        "id": "FV-01",
        "name": "frequency_baseline_equal",
        "scenario": "Frequency Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": EQUAL_COUNTS,  # All equal = all get 1.3x (tied rank 1)
        "expected_top": "journal",
        "expected_ranking": ["journal", "meditation", "gratitude", "quote"],
        "description": "Baseline: All equal counts (10 each) - base weights decide ranking"
    },
    {
        "id": "FV-02",
        "name": "frequency_meditation_most_used",
        "scenario": "Frequency Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": {"meditation": 50, "journal": 5, "gratitude": 3, "quote": 1},
        "expected_top": "journal",  # journal: 0.9*1.2*1.2=1.296, meditation: 0.8*1.2*1.3=1.248
        "expected_ranking": ["journal", "meditation", "gratitude", "quote"],
        "description": "Meditation most used (50): Can 1.3x freq flip meditation(0.8) over journal(0.9)?"
    },
    {
        "id": "FV-03",
        "name": "frequency_quote_most_used",
        "scenario": "Frequency Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": {"quote": 50, "journal": 5, "meditation": 3, "gratitude": 1},
        "expected_top": "journal",  # journal: 0.9*1.2*1.2=1.296, quote: 0.6*1.2*1.3=0.936, gratitude: 0.7*1.2*1.05=0.882
        "expected_ranking": ["journal", "meditation", "quote", "gratitude"],
        "description": "Quote most used (50): Extreme freq boost (1.3x) on lowest base (0.6) = 0.936, still below journal"
    },
    {
        "id": "FV-04",
        "name": "frequency_journal_dominant",
        "scenario": "Frequency Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": {"journal": 50, "meditation": 30, "gratitude": 10, "quote": 1},
        "expected_top": "journal",  # journal: 0.9*1.2*1.3=1.404, meditation: 0.8*1.2*1.2=1.152
        "expected_ranking": ["journal", "meditation", "gratitude", "quote"],
        "description": "Journal gets both highest base (0.9) AND freq (1.3x) - maximum lead"
    },
    {
        "id": "FV-05",
        "name": "frequency_gratitude_most_used",
        "scenario": "Frequency Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": {"gratitude": 50, "journal": 1, "meditation": 1, "quote": 1},
        "expected_top": "journal",  # journal: 0.9*1.2*1.1=1.188, gratitude: 0.7*1.2*1.3=1.092
        "expected_ranking": ["journal", "meditation", "gratitude", "quote"],
        "description": "Gratitude most used (50): Mid-tier base (0.7) with max freq vs top base (0.9)"
    },
    {
        "id": "FV-06",
        "name": "frequency_multiple_ties",
        "scenario": "Frequency Variation",
        "emotion": "Sad",
        "preferences": ALL_PREFERENCES_TRUE,
        "activity_counts": {"journal": 10, "meditation": 10, "gratitude": 50, "quote": 50},
        "expected_top": "journal",  # journal=meditation tied at 1.3x, gratitude=quote tied at 1.3x
        "expected_ranking": ["journal", "meditation", "gratitude", "quote"],
        "description": "Two tied at top freq (1.3x): journal=meditation=10, gratitude=quote=50"
    },
]


# =============================================================================
# Test Runner
# =============================================================================

def run_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case and return results."""
    emotion = test_case["emotion"]
    preferences = test_case["preferences"]
    activity_counts = test_case["activity_counts"]
    
    # Calculate raw scores (for transparency)
    raw_scores = calc_raw_scores(emotion, preferences, activity_counts)
    
    # Run suggestion engine
    try:
        ranked_activities, reasoning = suggest_activities(emotion, preferences, activity_counts)
        actual_ranking = [r["activity_type"] for r in ranked_activities]
        actual_top = actual_ranking[0] if actual_ranking else "ERROR"
        error = None
    except Exception as e:
        ranked_activities = []
        actual_ranking = []
        actual_top = "ERROR"
        error = str(e)
    
    # Determine pass/fail
    expected_top = test_case["expected_top"]
    expected_ranking = test_case["expected_ranking"]
    
    top_match = (actual_top == expected_top)
    ranking_match = (actual_ranking == expected_ranking)
    passed = top_match and ranking_match and error is None
    
    return {
        "id": test_case["id"],
        "name": test_case["name"],
        "scenario": test_case["scenario"],
        "description": test_case["description"],
        "emotion": emotion,
        "preferences": format_preferences(preferences),
        "activity_counts": format_counts(activity_counts),
        "raw_scores": format_raw_scores(raw_scores),
        "expected_top": expected_top,
        "actual_top": actual_top,
        "expected_ranking": " > ".join(expected_ranking),
        "actual_ranking": " > ".join(actual_ranking) if actual_ranking else "ERROR",
        "top_match": "Y" if top_match else "N",
        "ranking_match": "Y" if ranking_match else "N",
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
        "id", "name", "scenario", "description",
        "emotion", "preferences", "activity_counts", "raw_scores",
        "expected_top", "actual_top", "expected_ranking", "actual_ranking",
        "top_match", "ranking_match", "status", "error"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: List[Dict[str, Any]]):
    """Print summary table to console."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    
    print("\n" + "=" * 130)
    print("SUGGESTION ENGINE - MULTIPLIER STACKING TEST RESULTS")
    print("=" * 130)
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("\n" + "-" * 130)
    
    # Header
    print(f"{'ID':<8} {'Scenario':<18} {'Emotion':<8} {'Expected Top':<14} {'Actual Top':<14} {'Ranking OK':<12} {'Status':<8}")
    print("-" * 130)
    
    # Rows
    for r in results:
        status_icon = "[OK]" if r["status"] == "PASS" else "[X]"
        print(f"{r['id']:<8} {r['scenario']:<18} {r['emotion']:<8} {r['expected_top']:<14} {r['actual_top']:<14} {r['ranking_match']:<12} {status_icon} {r['status']}")
    
    print("-" * 130)
    print(f"\nPassed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    
    # Show failures in detail
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print("\n" + "=" * 130)
        print("FAILED TESTS - DETAILS")
        print("=" * 130)
        for r in failures:
            print(f"\n{r['id']}: {r['name']}")
            print(f"  Description: {r['description']}")
            print(f"  Raw Scores: {r['raw_scores']}")
            print(f"  Expected: {r['expected_ranking']}")
            print(f"  Actual:   {r['actual_ranking']}")
            if r['error']:
                print(f"  Error: {r['error']}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Running suggest_activities() multiplier stacking tests...")
    
    # Run tests
    results = run_all_tests()
    
    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "suggestion_multiplier_test_results.csv")
    save_csv(results, csv_path)
    print(f"\nCSV saved to: {csv_path}")
    
    # Print summary
    print_summary(results)
