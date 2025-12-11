#!/usr/bin/env python3
"""
Remote Demo Script for Fusion Service

This script allows you to test the fusion service remotely by providing
emotion:score inputs directly via the demo endpoint.

Usage:
    python testing/demo_fusion_remote.py
    
    Or set environment variables:
    export FUSION_SERVICE_URL=https://your-service-url.run.app
    export DEV_USER_ID=your-user-uuid
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


def check_fusion_service(url: str):
    """Check if fusion service is accessible."""
    try:
        response = requests.get(f"{url}/emotion/health", timeout=5)
        if response.status_code == 200:
            return True, None
        else:
            return False, f"Service returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to service"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, str(e)


def get_latest_emotion_log(user_id: str):
    """Get the latest emotion log entry before our test."""
    try:
        return database.get_latest_emotion_log(user_id)
    except Exception as e:
        print(f"  ⚠ Error fetching latest emotion log: {e}")
        return None


def parse_signal_input(input_str: str) -> str:
    """
    Parse and validate signal input string.
    Returns the input string if valid, or None if invalid.
    """
    if not input_str or input_str.strip() == "":
        return None
    
    # Basic validation: check format
    parts = input_str.split(",")
    for part in parts:
        part = part.strip()
        if ":" not in part:
            return None
        emotion, confidence = part.split(":", 1)
        try:
            conf_float = float(confidence.strip())
            if conf_float < 0.0 or conf_float > 1.0:
                return None
        except ValueError:
            return None
    
    return input_str.strip()


def demo_fusion_snapshot(service_url: str, user_id: str, signals: dict):
    """
    Call the demo fusion snapshot endpoint.
    
    Args:
        service_url: Base URL of fusion service
        user_id: User ID
        signals: Dictionary with modality keys and signal strings
    
    Returns:
        Response dictionary or None on failure
    """
    print_section("DEMO FUSION SNAPSHOT")
    
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
    print("\n2. Preparing demo request...")
    request_data = {
        "user_id": user_id,
        "signals": signals
    }
    
    print(f"   Signals:")
    for modality, signal_str in signals.items():
        if signal_str:
            print(f"     {modality}: {signal_str}")
        else:
            print(f"     {modality}: (empty)")
    
    # Call fusion service demo endpoint
    print("\n3. Calling fusion service demo endpoint...")
    print(f"   URL: {service_url}/emotion/snapshot/demo")
    
    try:
        response = requests.post(
            f"{service_url}/emotion/snapshot/demo",
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
            print("\n4. Fusion Result:")
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
        elif response.status_code == 403:
            print(f"\n   ✗ Demo mode not enabled")
            print(f"   Error: {response.json().get('detail', 'Unknown error')}")
            print(f"\n   💡 To enable demo mode, set DEMO_MODE_ENABLED=true on the fusion service")
            return None
        else:
            print(f"\n   ✗ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"   Error text: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n   ✗ Connection error: Could not connect to {service_url}")
        print("   Make sure the fusion service is running and accessible")
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


def main():
    """Main demo function."""
    print("\n" + "=" * 80)
    print("FUSION SERVICE REMOTE DEMO")
    print("=" * 80)
    print(f"User ID: {DEV_USER_ID}")
    print(f"Fusion Service URL: {FUSION_SERVICE_URL}")
    
    # Check service accessibility
    print("\nChecking fusion service accessibility...")
    is_accessible, error_msg = check_fusion_service(FUSION_SERVICE_URL)
    if not is_accessible:
        print(f"  ✗ Service not accessible: {error_msg}")
        print("\nPlease ensure:")
        print("  1. Fusion service is running")
        print("  2. DEMO_MODE_ENABLED=true is set on the service")
        print("  3. Service URL is correct")
        print("\nContinue anyway? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice != 'y':
                print("Exiting...")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
    else:
        print("  ✓ Service is accessible")
    
    # Prompt for signals
    print("\n" + "=" * 80)
    print("ENTER SIGNALS")
    print("=" * 80)
    print("\nEnter emotion:score signals for each modality.")
    print("Format: emotion:confidence (case-insensitive)")
    print("Multiple signals: emotion1:conf1,emotion2:conf2")
    print("Examples: sad:0.8  or  sad:0.8,happy:0.6")
    print("Press Enter to skip a modality (empty signals)\n")
    
    signals = {}
    
    # Prompt for speech signals
    print("Speech signals: ", end="", flush=True)
    try:
        speech_input = input().strip()
        speech_parsed = parse_signal_input(speech_input)
        signals["speech"] = speech_parsed if speech_parsed else ""
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted")
        signals["speech"] = ""
    
    # Prompt for face signals
    print("Face signals: ", end="", flush=True)
    try:
        face_input = input().strip()
        face_parsed = parse_signal_input(face_input)
        signals["face"] = face_parsed if face_parsed else ""
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted")
        signals["face"] = ""
    
    # Prompt for vitals signals
    print("Vitals signals: ", end="", flush=True)
    try:
        vitals_input = input().strip()
        vitals_parsed = parse_signal_input(vitals_input)
        signals["vitals"] = vitals_parsed if vitals_parsed else ""
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted")
        signals["vitals"] = ""
    
    # Check if any signals provided
    if not any(signals.values()):
        print("\n⚠ No signals provided. Exiting...")
        return
    
    # Call demo endpoint
    result = demo_fusion_snapshot(FUSION_SERVICE_URL, DEV_USER_ID, signals)
    
    if result:
        expected_emotion = result.get('emotion_label')
        verify_database_write(DEV_USER_ID, expected_emotion)
        
        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print("✓ Demo endpoint called successfully")
        print("✓ Fusion result received")
        print("✓ Database entry verified")
        print("\nDemo completed successfully!")
    else:
        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print("✗ Demo failed - no fusion result received")
        print("\nPossible issues:")
        print("  - Demo mode not enabled (DEMO_MODE_ENABLED=true)")
        print("  - Invalid signal format")
        print("  - Service connectivity issues")
    
    # Ask if user wants to run another demo
    print("\n" + "=" * 80)
    print("Run another demo? (y/n): ", end="")
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
        print("\n\nDemo interrupted by user")
        sys.exit(0)

