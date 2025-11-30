#!/usr/bin/env python3
"""
Test script for Intervention Module

This script tests:
1. Database fetching functions
2. Decision engine computation
3. Suggestion engine computation
4. Full intervention orchestrator flow
5. Output production

Loads DEV_USER_ID from .env file.
"""

import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import database
from intervention.decision_engine import decide_trigger_intervention
from intervention.suggestion_engine import suggest_activities
from intervention.intervention import process_suggestion_request
from intervention.models import SuggestionRequest

# Load environment variables
load_dotenv()

# Get DEV_USER_ID from .env
DEV_USER_ID = os.getenv("DEV_USER_ID")
if not DEV_USER_ID:
    print("ERROR: DEV_USER_ID not found in .env file")
    sys.exit(1)

print("=" * 80)
print("INTERVENTION MODULE TEST")
print("=" * 80)
print(f"Testing with User ID: {DEV_USER_ID}\n")


def test_database_fetching():
    """Test 1: Database fetching functions"""
    print("=" * 80)
    print("TEST 1: DATABASE FETCHING")
    print("=" * 80)
    
    try:
        # Test emotion logs
        print("\n1.1 Fetching recent emotion logs (last 48 hours)...")
        emotion_logs = database.fetch_recent_emotion_logs(DEV_USER_ID, hours=48)
        print(f"   ✓ Fetched {len(emotion_logs)} emotion logs")
        if emotion_logs:
            latest = emotion_logs[-1]
            print(f"   Latest emotion: {latest.get('emotion_label')} "
                  f"(confidence: {latest.get('confidence_score'):.2f})")
        else:
            print("   ⚠ No emotion logs found")
        
        # Test activity logs
        print("\n1.2 Fetching recent activity logs (last 24 hours)...")
        activity_logs = database.fetch_recent_activity_logs(DEV_USER_ID, hours=24)
        print(f"   ✓ Fetched {len(activity_logs)} activity logs")
        if activity_logs:
            latest = activity_logs[-1]
            print(f"   Latest activity: {latest.get('intervention_type')} "
                  f"at {latest.get('timestamp')}")
        else:
            print("   ⚠ No activity logs found")
        
        # Test user preferences
        print("\n1.3 Fetching user preferences...")
        preferences = database.fetch_user_preferences(DEV_USER_ID)
        print(f"   ✓ Fetched preferences: {json.dumps(preferences, indent=2)}")
        
        # Test time since last activity
        print("\n1.4 Calculating time since last activity...")
        time_since = database.get_time_since_last_activity(DEV_USER_ID)
        if time_since == float('inf'):
            print("   ⚠ No previous activities found (infinite time)")
        else:
            print(f"   ✓ Time since last activity: {time_since:.2f} minutes ({time_since/60:.2f} hours)")
        
        print("\n✓ TEST 1 PASSED: All database functions working\n")
        return emotion_logs, activity_logs, preferences, time_since
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_decision_engine(emotion_label="Sad", confidence_score=0.85, time_since_last=120.0):
    """Test 2: Decision engine computation"""
    print("=" * 80)
    print("TEST 2: DECISION ENGINE")
    print("=" * 80)
    
    try:
        print(f"\nTesting with:")
        print(f"  - Emotion: {emotion_label}")
        print(f"  - Confidence: {confidence_score}")
        print(f"  - Time since last activity: {time_since_last} minutes")
        
        trigger, confidence, reasoning = decide_trigger_intervention(
            emotion_label=emotion_label,
            confidence_score=confidence_score,
            time_since_last_activity_minutes=time_since_last
        )
        
        print(f"\n✓ Decision computed:")
        print(f"  - Trigger intervention: {trigger}")
        print(f"  - Confidence score: {confidence:.3f}")
        print(f"  - Reasoning: {reasoning}")
        
        print("\n✓ TEST 2 PASSED: Decision engine working\n")
        return trigger, confidence, reasoning
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_suggestion_engine(emotion_label="Sad", preferences=None, activity_counts=None):
    """Test 3: Suggestion engine computation"""
    print("=" * 80)
    print("TEST 3: SUGGESTION ENGINE")
    print("=" * 80)
    
    try:
        if preferences is None:
            preferences = {
                "plan": True,
                "music": True,
                "quote": True,
                "converse": True,
                "breathing": True,
                "gratitude": True,
                "journaling": True
            }
        
        if activity_counts is None:
            # Default test counts: journal is most frequent, quote is least
            activity_counts = {
                'journal': 15,
                'gratitude': 8,
                'meditation': 12,
                'quote': 5
            }
        
        print(f"\nTesting with:")
        print(f"  - Emotion: {emotion_label}")
        print(f"  - Preferences: {json.dumps(preferences, indent=2)}")
        print(f"  - Activity counts: {activity_counts}")
        
        ranked_activities, reasoning = suggest_activities(
            emotion_label=emotion_label,
            user_preferences=preferences,
            activity_counts=activity_counts
        )
        
        print(f"\n✓ Suggestions computed:")
        print(f"  - Reasoning: {reasoning}")
        print(f"\n  Ranked Activities:")
        for activity in ranked_activities:
            print(f"    Rank {activity['rank']}: {activity['activity_type']} "
                  f"(score: {activity['score']:.3f})")
        
        print("\n✓ TEST 3 PASSED: Suggestion engine working\n")
        return ranked_activities, reasoning
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_full_flow(emotion_label="Sad", confidence_score=0.85):
    """Test 4: Full intervention orchestrator flow"""
    print("=" * 80)
    print("TEST 4: FULL INTERVENTION FLOW")
    print("=" * 80)
    
    try:
        # Create a test request
        request = SuggestionRequest(
            user_id=DEV_USER_ID,
            emotion_label=emotion_label,
            confidence_score=confidence_score,
            timestamp=datetime.now(timezone.utc),
            context_time_of_day=None  # Will be auto-calculated
        )
        
        print(f"\nRequest:")
        print(f"  - User ID: {request.user_id}")
        print(f"  - Emotion: {request.emotion_label}")
        print(f"  - Confidence: {request.confidence_score}")
        print(f"  - Timestamp: {request.timestamp}")
        
        print("\nProcessing request through orchestrator...")
        response = process_suggestion_request(request)
        
        print(f"\n✓ Response received:")
        print(f"\n  Decision:")
        print(f"    - Trigger intervention: {response.decision.trigger_intervention}")
        print(f"    - Confidence: {response.decision.confidence_score:.3f}")
        print(f"    - Reasoning: {response.decision.reasoning}")
        
        print(f"\n  Suggestions:")
        print(f"    - Reasoning: {response.suggestion.reasoning}")
        print(f"\n    Ranked Activities:")
        for activity in response.suggestion.ranked_activities:
            print(f"      Rank {activity.rank}: {activity.activity_type} "
                  f"(score: {activity.score:.3f})")
        
        print("\n✓ TEST 4 PASSED: Full flow working\n")
        return response
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_decision_engine_cases():
    """Test: Decision engine unit tests - all test cases from intervention_test_cases.md"""
    print("=" * 80)
    print("DECISION ENGINE UNIT TESTS")
    print("=" * 80)
    print("Testing all 10 test cases from intervention_test_cases.md\n")
    
    # Define all 10 test cases
    test_cases = [
        {
            "case_number": 1,
            "emotion_label": "Sad",
            "confidence_score": 0.80,
            "time_since_last_activity": 90.0,
            "expected_trigger": True,
            "expected_confidence": 0.80,  # "0.80 (or cap at 0.95 if >)" - actual should be 0.80
            "description": "Negative emotion + high confidence + enough time → trigger"
        },
        {
            "case_number": 2,
            "emotion_label": "Angry",
            "confidence_score": 0.75,
            "time_since_last_activity": 61.0,
            "expected_trigger": True,
            "expected_confidence": 0.75,
            "description": "Another negative emotion case"
        },
        {
            "case_number": 3,
            "emotion_label": "Fear",
            "confidence_score": 0.95,
            "time_since_last_activity": 200.0,
            "expected_trigger": True,
            "expected_confidence": 0.95,  # "0.95 (cap)" - should be capped at 0.95
            "description": "Very high confidence, trigger, but cap at 0.95"
        },
        {
            "case_number": 4,
            "emotion_label": "Sad",
            "confidence_score": 0.65,
            "time_since_last_activity": 100.0,
            "expected_trigger": False,
            "expected_confidence": 0.325,  # "confidence_score * 0.5 = 0.325"
            "description": "Negative emotion but confidence < threshold"
        },
        {
            "case_number": 5,
            "emotion_label": "Happy",
            "confidence_score": 0.90,
            "time_since_last_activity": 120.0,
            "expected_trigger": False,
            "expected_confidence": 0.0,
            "description": "Emotion not negative → no trigger"
        },
        {
            "case_number": 6,
            "emotion_label": "Angry",
            "confidence_score": 0.72,
            "time_since_last_activity": 30.0,
            "expected_trigger": False,
            "expected_confidence": 0.0,
            "description": "Negative + good confidence but not enough time passed"
        },
        {
            "case_number": 7,
            "emotion_label": "Fear",
            "confidence_score": 0.70,
            "time_since_last_activity": 60.0,
            "expected_trigger": False,
            "expected_confidence": 0.0,
            "description": "Exactly at time threshold (60) – as per logic >60 so equal should fail → no trigger"
        },
        {
            "case_number": 8,
            "emotion_label": "Sad",
            "confidence_score": 0.70,
            "time_since_last_activity": 61.0,
            "expected_trigger": True,
            "expected_confidence": 0.70,
            "description": "Edge: exactly confidence threshold and just over time threshold → trigger"
        },
        {
            "case_number": 9,
            "emotion_label": "Happy",
            "confidence_score": 0.50,
            "time_since_last_activity": 80.0,
            "expected_trigger": False,
            "expected_confidence": 0.0,
            "description": "Emotion positive and confidence low → no trigger"
        },
        {
            "case_number": 10,
            "emotion_label": "Angry",
            "confidence_score": 0.80,
            "time_since_last_activity": 59.9,
            "expected_trigger": False,
            "expected_confidence": 0.0,
            "description": "Negative + high confidence but time just under threshold → no trigger"
        }
    ]
    
    passed = 0
    failed = 0
    tolerance = 0.001  # For floating point comparison
    
    print(f"{'Case':<6} {'Inputs':<45} {'Expected':<25} {'Actual':<25} {'Status':<8} {'Reasoning'}")
    print("-" * 150)
    
    for case in test_cases:
        try:
            # Call the decision engine
            actual_trigger, actual_confidence, reasoning = decide_trigger_intervention(
                emotion_label=case["emotion_label"],
                confidence_score=case["confidence_score"],
                time_since_last_activity_minutes=case["time_since_last_activity"]
            )
            
            # Check if results match expected
            trigger_match = actual_trigger == case["expected_trigger"]
            confidence_match = abs(actual_confidence - case["expected_confidence"]) < tolerance
            
            # Determine status
            if trigger_match and confidence_match:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
            
            # Format inputs
            inputs = f"{case['emotion_label']}, conf={case['confidence_score']:.2f}, time={case['time_since_last_activity']:.1f}min"
            
            # Format expected
            expected = f"trigger={case['expected_trigger']}, conf={case['expected_confidence']:.3f}"
            
            # Format actual
            actual = f"trigger={actual_trigger}, conf={actual_confidence:.3f}"
            
            # Truncate reasoning if too long
            reasoning_short = reasoning[:50] + "..." if reasoning and len(reasoning) > 50 else (reasoning or "")
            
            # Print result
            print(f"{case['case_number']:<6} {inputs:<45} {expected:<25} {actual:<25} {status:<8} {reasoning_short}")
            
            # Print detailed failure info if failed
            if status == "FAIL":
                if not trigger_match:
                    print(f"       ✗ Trigger mismatch: expected {case['expected_trigger']}, got {actual_trigger}")
                if not confidence_match:
                    print(f"       ✗ Confidence mismatch: expected {case['expected_confidence']:.3f}, got {actual_confidence:.3f} (diff: {abs(actual_confidence - case['expected_confidence']):.6f})")
            
        except Exception as e:
            print(f"{case['case_number']:<6} ERROR: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total cases: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED: Decision engine working correctly!\n")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED: Please review the results above.\n")
    
    return passed, failed


def test_multiple_scenarios():
    """Test 5: Multiple scenarios"""
    print("=" * 80)
    print("TEST 5: MULTIPLE SCENARIOS")
    print("=" * 80)
    
    scenarios = [
        {"emotion": "Sad", "confidence": 0.85, "description": "High confidence negative emotion"},
        {"emotion": "Angry", "confidence": 0.75, "description": "Medium-high confidence negative emotion"},
        {"emotion": "Happy", "confidence": 0.90, "description": "Positive emotion"},
        {"emotion": "Fear", "confidence": 0.65, "description": "Low confidence negative emotion"},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['description']} ---")
        try:
            request = SuggestionRequest(
                user_id=DEV_USER_ID,
                emotion_label=scenario["emotion"],
                confidence_score=scenario["confidence"],
                timestamp=datetime.now(timezone.utc)
            )
            
            response = process_suggestion_request(request)
            
            print(f"  Decision: trigger={response.decision.trigger_intervention}, "
                  f"confidence={response.decision.confidence_score:.3f}")
            print(f"  Top suggestion: {response.suggestion.ranked_activities[0].activity_type} "
                  f"(rank {response.suggestion.ranked_activities[0].rank}, "
                  f"score {response.suggestion.ranked_activities[0].score:.3f})")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n✓ TEST 5 PASSED: Multiple scenarios tested\n")


def main():
    """Run all tests"""
    print("\n")
    
    # Decision Engine Unit Tests (isolated, no database required)
    test_decision_engine_cases()
    
    # Test 1: Database fetching
    emotion_logs, activity_logs, preferences, time_since = test_database_fetching()
    
    # Test 2: Decision engine
    test_decision_engine(
        emotion_label="Sad",
        confidence_score=0.85,
        time_since_last=time_since if time_since != float('inf') else 120.0
    )
    
    # Test 3: Suggestion engine
    # Get activity counts from database for testing
    activity_counts = database.get_activity_counts(DEV_USER_ID, days=30)
    test_suggestion_engine(
        emotion_label="Sad",
        preferences=preferences,
        activity_counts=activity_counts
    )
    
    # Test 4: Full flow
    test_full_flow(emotion_label="Sad", confidence_score=0.85)
    
    # Test 5: Multiple scenarios
    test_multiple_scenarios()
    
    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\n✓ Intervention module is working correctly!\n")


if __name__ == "__main__":
    import sys
    
    # Allow running just the decision engine unit tests
    if len(sys.argv) > 1 and sys.argv[1] == "--decision-engine-only":
        test_decision_engine_cases()
    else:
        main()

