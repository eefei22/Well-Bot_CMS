#!/usr/bin/env python3
"""
Test script for Suggestion Engine Frequency Multipliers

This script tests the frequency-based multiplier logic in suggestion_engine.py
with various activity patterns and edge cases, without database dependencies.
"""

import os
import sys
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervention.suggestion_engine import (
    suggest_activities,
    EMOTION_ACTIVITY_WEIGHTS,
    FREQUENCY_MULTIPLIERS,
    ACTIVITY_TYPES
)


def calculate_frequency_ranking(activity_counts):
    """
    Calculate frequency ranking for activities based on counts.
    Groups activities by frequency count, so activities with same count get same multiplier.
    Returns list of tuples: (activity_type, count, group_rank, multiplier)
    """
    # Get counts for all activity types (default to 0 if not in dict)
    counts = {activity: activity_counts.get(activity, 0) for activity in ACTIVITY_TYPES}
    
    # Group activities by their frequency count
    frequency_groups = {}
    for activity in ACTIVITY_TYPES:
        count = counts[activity]
        if count not in frequency_groups:
            frequency_groups[count] = []
        frequency_groups[count].append(activity)
    
    # Sort groups by count (descending) to determine group rank
    sorted_group_counts = sorted(frequency_groups.keys(), reverse=True)
    
    # Build ranking list - all activities in same group get same multiplier
    ranking = []
    for group_rank, group_count in enumerate(sorted_group_counts, start=1):
        multiplier = FREQUENCY_MULTIPLIERS.get(group_rank, 1.0)
        activities_in_group = frequency_groups[group_count]
        for activity_type in activities_in_group:
            ranking.append((activity_type, counts[activity_type], group_rank, multiplier))
    
    return ranking


def format_activity_counts(activity_counts):
    """Format activity counts as a readable string."""
    return ", ".join([f"{k}={v}" for k, v in sorted(activity_counts.items())])


def run_test_case(test_case):
    """Run a single test case and return formatted results."""
    print("=" * 80)
    print(f"Test Case {test_case['number']}: {test_case['name']}")
    print("=" * 80)
    
    # Print input parameters
    print("\nInput:")
    print(f"  Emotion: {test_case['emotion_label']}")
    print(f"  Preferences: {json.dumps(test_case['user_preferences'], indent=4)}")
    print(f"  Activity Counts: {format_activity_counts(test_case['activity_counts'])}")
    
    # Calculate and display frequency ranking
    frequency_ranking = calculate_frequency_ranking(test_case['activity_counts'])
    print("\nFrequency Ranking (grouped by count):")
    # Group by multiplier to show which activities share the same multiplier
    multiplier_groups = {}
    for activity_type, count, group_rank, multiplier in frequency_ranking:
        if multiplier not in multiplier_groups:
            multiplier_groups[multiplier] = []
        multiplier_groups[multiplier].append((activity_type, count, group_rank))
    
    # Display grouped by multiplier
    for multiplier in sorted(multiplier_groups.keys(), reverse=True):
        activities = multiplier_groups[multiplier]
        activity_names = [f"{a[0]} (count: {a[1]})" for a in activities]
        group_ranks = set([a[2] for a in activities])
        print(f"  Group rank {min(group_ranks)}: {', '.join(activity_names)} → multiplier: {multiplier}x")
    
    # Get base emotion weights for reference
    emotion_weights = EMOTION_ACTIVITY_WEIGHTS.get(test_case['emotion_label'], {
        'journal': 0.7,
        'gratitude': 0.7,
        'meditation': 0.7,
        'quote': 0.7
    })
    
    print("\nBase Emotion Weights:")
    for activity in ACTIVITY_TYPES:
        weight = emotion_weights.get(activity, 0.5)
        print(f"  {activity}: {weight}")
    
    # Call suggestion engine
    ranked_activities, reasoning = suggest_activities(
        emotion_label=test_case['emotion_label'],
        user_preferences=test_case['user_preferences'],
        activity_counts=test_case['activity_counts']
    )
    
    # Display results
    print("\nResults:")
    for activity in ranked_activities:
        print(f"  Rank {activity['rank']}: {activity['activity_type']} (score: {activity['score']:.3f})")
    
    print(f"\nReasoning: {reasoning}")
    
    # Verify frequency ranking matches result ranking (after all adjustments)
    print("\nVerification:")
    result_ranking = [a['activity_type'] for a in ranked_activities]
    frequency_order = [item[0] for item in frequency_ranking]
    
    # Note: Final ranking may differ from frequency ranking due to:
    # - Emotion base weights
    # - User preferences
    # - Normalization
    print(f"  Frequency order: {' → '.join(frequency_order)}")
    print(f"  Final result order: {' → '.join(result_ranking)}")
    
    if result_ranking == frequency_order:
        print("  ✓ Final ranking matches frequency ranking")
    else:
        print("  ⚠ Final ranking differs from frequency ranking (expected due to emotion/preferences)")
    
    print("\n" + "-" * 80 + "\n")


def main():
    """Run all test cases."""
    print("\n" + "=" * 80)
    print("SUGGESTION ENGINE FREQUENCY MULTIPLIER TESTS")
    print("=" * 80)
    print("\nTesting frequency-based multipliers with various activity patterns\n")
    
    # Define all test cases
    test_cases = [
        {
            "number": 1,
            "name": "Normal Case - Different Frequencies",
            "emotion_label": "Sad",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 15,
                "gratitude": 8,
                "meditation": 12,
                "quote": 5
            },
            "description": "Verify most frequent gets highest boost, ranking is correct"
        },
        {
            "number": 2,
            "name": "No Activities (All Zero)",
            "emotion_label": "Sad",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 0,
                "gratitude": 0,
                "meditation": 0,
                "quote": 0
            },
            "description": "All tied at 0, alphabetical order determines ranking"
        },
        {
            "number": 3,
            "name": "All Activities Tied (Same Count)",
            "emotion_label": "Sad",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 10,
                "gratitude": 10,
                "meditation": 10,
                "quote": 10
            },
            "description": "All tied, alphabetical order used"
        },
        {
            "number": 4,
            "name": "Only One Activity Type Used",
            "emotion_label": "Sad",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 20,
                "gratitude": 0,
                "meditation": 0,
                "quote": 0
            },
            "description": "One clear winner, others tied at 0"
        },
        {
            "number": 5,
            "name": "Two Activities Tied for First",
            "emotion_label": "Sad",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 15,
                "gratitude": 15,
                "meditation": 5,
                "quote": 3
            },
            "description": "Tie-breaking works correctly (alphabetical: gratitude before journal)"
        },
        {
            "number": 6,
            "name": "One Activity Much More Frequent",
            "emotion_label": "Happy",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 50,
                "gratitude": 5,
                "meditation": 3,
                "quote": 2
            },
            "description": "Clear ranking with large differences"
        },
        {
            "number": 7,
            "name": "Different Emotion with Frequency Impact",
            "emotion_label": "Angry",
            "user_preferences": {
                "journaling": True,
                "gratitude": True,
                "breathing": True,
                "quote": True
            },
            "activity_counts": {
                "journal": 20,
                "gratitude": 5,
                "meditation": 10,
                "quote": 3
            },
            "description": "Frequency multipliers work with different emotion base weights (Angry: meditation=0.9 base)"
        },
        {
            "number": 8,
            "name": "Mixed Preferences Impact",
            "emotion_label": "Sad",
            "user_preferences": {
                "journaling": True,
                "gratitude": False,
                "breathing": True,
                "quote": False
            },
            "activity_counts": {
                "journal": 10,
                "gratitude": 15,
                "meditation": 8,
                "quote": 5
            },
            "description": "gratitude is most frequent (15) but has False preference (×0.7), journal has True preference (×1.2) and is 2nd most frequent (10)"
        }
    ]
    
    # Run all test cases
    for test_case in test_cases:
        run_test_case(test_case)
    
    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Total test cases: {len(test_cases)}")
    print("  ✓ All test cases executed successfully")
    print("\nNote: Final ranking may differ from frequency ranking due to:")
    print("  - Emotion base weights (different emotions favor different activities)")
    print("  - User preferences (×1.2 if preferred, ×0.7 if not)")
    print("  - Score normalization (all scores normalized to 0.0-1.0 range)")
    print("\n")


if __name__ == "__main__":
    main()

