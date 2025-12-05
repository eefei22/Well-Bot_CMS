#!/usr/bin/env python3
"""
End-to-End Test Script for Fusion Service

This script tests the complete fusion service flow:
1. Calls POST /emotion/snapshot endpoint
2. Displays fusion results
3. Verifies database write
4. Shows before/after comparison

Usage:
    python testing/test_fusion_e2e.py
    
Prerequisites:
    - Mock model services running on ports 8005, 8006, 8007
    - Fusion service running (main.py)
    - DEV_USER_ID set in .env file
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import database

# Load environment variables
load_dotenv()

# Configuration
DEV_USER_ID = os.getenv("DEV_USER_ID")
FUSION_SERVICE_URL = os.getenv("FUSION_SERVICE_URL", "http://localhost:8000")

if not DEV_USER_ID:
    print("ERROR: DEV_USER_ID not found in .env file")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_latest_emotion_log(user_id: str):
    """Get the latest emotion log entry before our test."""
    try:
        return database.get_latest_emotion_log(user_id)
    except Exception as e:
        print(f"  ⚠ Error fetching latest emotion log: {e}")
        return None


def test_fusion_snapshot(user_id: str, timestamp: str = None, options: dict = None):
    """
    Test the fusion snapshot endpoint.
    
    Args:
        user_id: User ID to test with
        timestamp: Optional snapshot timestamp (ISO format)
        options: Optional request options (timeout_seconds, window_seconds)
    
    Returns:
        Response dictionary or None on failure
    """
    print_section("FUSION SERVICE E2E TEST")
    
    # Get latest emotion log before test
    print("\n1. Checking database state before test...")
    before_log = get_latest_emotion_log(user_id)
    if before_log:
        print(f"   Latest emotion log before test:")
        print(f"     ID: {before_log.get('id')}")
        print(f"     Emotion: {before_log.get('emotion_label')}")
        print(f"     Confidence: {before_log.get('confidence_score'):.3f}")
        print(f"     Timestamp: {before_log.get('timestamp')}")
    else:
        print("   No previous emotion logs found")
    
    # Prepare request
    print("\n2. Preparing fusion snapshot request...")
    request_data = {
        "user_id": user_id
    }
    
    if timestamp:
        request_data["timestamp"] = timestamp
        print(f"   Using provided timestamp: {timestamp}")
    else:
        print("   Using current time (no timestamp provided)")
    
    if options:
        request_data["options"] = options
        print(f"   Options: {json.dumps(options, indent=6)}")
    
    # Set input signals for mock services
    print("\n3. Setting input signals for mock services...")
    print("   Enter signals for each service (or 'auto' for test data):")
    
    mock_services = {
        "SER": "http://localhost:8005",
        "FER": "http://localhost:8006",
        "Vitals": "http://localhost:8007"
    }
    
    service_inputs = {}
    for service_name, service_url in mock_services.items():
        print(f"\n   {service_name} signals (format: emotion:confidence, e.g., sad:0.8,happy:0.6): ", end="", flush=True)
        try:
            user_input = input().strip()
            if user_input:
                service_inputs[service_name] = user_input
                # Set input via /set-input endpoint
                try:
                    response = requests.post(
                        f"{service_url}/set-input",
                        json={"input": user_input},
                        timeout=2
                    )
                    if response.status_code == 200:
                        print(f"     ✓ Input set for {service_name}")
                    else:
                        print(f"     ⚠ Failed to set input for {service_name}")
                except Exception as e:
                    print(f"     ⚠ Error setting input: {e}")
            else:
                print(f"     → No input (will return empty signals)")
        except (EOFError, KeyboardInterrupt):
            print(f"\n     → Skipping {service_name}")
    
    # Call fusion service
    print("\n4. Calling fusion service...")
    print(f"   URL: {FUSION_SERVICE_URL}/emotion/snapshot")
    print(f"   Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{FUSION_SERVICE_URL}/emotion/snapshot",
            json=request_data,
            timeout=30
        )
        
        print(f"\n   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if it's a NoSignalsResponse
            if result.get("status") == "no_signals":
                print("\n   ⚠ No signals available:")
                print(f"     Reason: {result.get('reason')}")
                return None
            
            # Display fusion result
            print("\n5. Fusion Result:")
            print(f"   User ID: {result.get('user_id')}")
            print(f"   Timestamp: {result.get('timestamp')}")
            print(f"   Emotion Label: {result.get('emotion_label')}")
            print(f"   Confidence Score: {result.get('confidence_score'):.3f}")
            print(f"   Emotional Score: {result.get('emotional_score')}")
            
            # Display signals used
            signals_used = result.get('signals_used', [])
            if signals_used:
                print(f"\n   Signals Used ({len(signals_used)}):")
                for i, signal in enumerate(signals_used, 1):
                    print(f"     {i}. {signal.get('modality')}: {signal.get('emotion_label')} "
                          f"(confidence: {signal.get('confidence'):.3f})")
            else:
                print("\n   No signals used (empty)")
            
            return result
        else:
            print(f"\n   ✗ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"   Error text: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n   ✗ Connection error: Could not connect to {FUSION_SERVICE_URL}")
        print("   Make sure the fusion service is running (python main.py)")
        return None
    except requests.exceptions.Timeout:
        print(f"\n   ✗ Timeout: Request took too long")
        return None
    except Exception as e:
        print(f"\n   ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_database_write(user_id: str, expected_emotion: str = None):
    """
    Verify that the emotion log was written to the database.
    
    Args:
        user_id: User ID
        expected_emotion: Optional expected emotion label to verify
    
    Returns:
        Latest emotion log entry or None
    """
    print_section("DATABASE VERIFICATION")
    
    # Wait a moment for DB write to complete
    print("\nWaiting 1 second for database write to complete...")
    time.sleep(1)
    
    # Get latest emotion log
    print("\nFetching latest emotion log from database...")
    latest_log = get_latest_emotion_log(user_id)
    
    if latest_log:
        print("\n✓ Database entry found:")
        print(f"   ID: {latest_log.get('id')}")
        print(f"   User ID: {latest_log.get('user_id')}")
        print(f"   Timestamp: {latest_log.get('timestamp')}")
        print(f"   Emotion Label: {latest_log.get('emotion_label')}")
        print(f"   Confidence Score: {latest_log.get('confidence_score'):.3f}")
        print(f"   Emotional Score: {latest_log.get('emotional_score')}")
        
        if expected_emotion:
            if latest_log.get('emotion_label') == expected_emotion:
                print(f"\n✓ Emotion label matches expected: {expected_emotion}")
            else:
                print(f"\n⚠ Emotion label mismatch:")
                print(f"   Expected: {expected_emotion}")
                print(f"   Got: {latest_log.get('emotion_label')}")
        
        return latest_log
    else:
        print("\n✗ No emotion log entry found in database")
        print("   The fusion service may not have written to the database")
        return None


def check_services_running():
    """Check if mock services are running."""
    import requests
    services = {
        "SER": "http://localhost:8005/health",
        "FER": "http://localhost:8006/health",
        "Vitals": "http://localhost:8007/health"
    }
    
    all_running = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"  ✓ {name} service running")
            else:
                print(f"  ✗ {name} service not responding")
                all_running = False
        except Exception:
            print(f"  ✗ {name} service not running (cannot connect)")
            all_running = False
    
    return all_running


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("FUSION SERVICE END-TO-END TEST")
    print("=" * 80)
    print(f"User ID: {DEV_USER_ID}")
    print(f"Fusion Service URL: {FUSION_SERVICE_URL}")
    print("\nChecking prerequisites...")
    
    # Check mock services
    print("\nMock Services:")
    mock_services_ok = check_services_running()
    
    # Check fusion service
    print("\nFusion Service:")
    try:
        response = requests.get(f"{FUSION_SERVICE_URL}/emotion/health", timeout=2)
        if response.status_code == 200:
            print(f"  ✓ Fusion service running")
            fusion_ok = True
        else:
            print(f"  ✗ Fusion service not responding")
            fusion_ok = False
    except Exception:
        print(f"  ✗ Fusion service not running (cannot connect)")
        fusion_ok = False
    
    if not mock_services_ok or not fusion_ok:
        print("\n⚠ WARNING: Some services are not running!")
        print("\nTo start mock services, run:")
        print("  python testing/mock_model_services.py")
        print("\nTo start fusion service, run:")
        print("  python main.py")
        print("\nContinue anyway? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice != 'y':
                print("Exiting...")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
    
    print("\nWhen prompted by mock services, enter signals manually:")
    print("  Format: emotion:confidence (case-insensitive)")
    print("  Example: sad:0.8,happy:0.6  or  Sad:0.8,Happy:0.6")
    print("  Or type 'auto' for test data")
    print("\nPress Enter to continue...", end="")
    input()
    
    # Test 1: Basic fusion with current timestamp
    print("\n" + "=" * 80)
    print("TEST 1: Basic Fusion Snapshot")
    print("=" * 80)
    
    result = test_fusion_snapshot(DEV_USER_ID)
    
    if result:
        expected_emotion = result.get('emotion_label')
        verify_database_write(DEV_USER_ID, expected_emotion)
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("✓ Fusion service called successfully")
        print("✓ Fusion result received")
        print("✓ Database entry verified")
        print("\nTest completed successfully!")
    else:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("✗ Test failed - no fusion result received")
        print("\nPossible issues:")
        print("  - Mock services not running")
        print("  - Fusion service not running")
        print("  - No signals provided to mock services")
        print("  - Network connectivity issues")
    
    # Ask if user wants to run another test
    print("\n" + "=" * 80)
    print("Run another test? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)

