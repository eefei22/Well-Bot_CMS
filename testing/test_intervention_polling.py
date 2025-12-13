#!/usr/bin/env python3
"""
Edge Device Polling Simulation Script

This script simulates an edge device polling the intervention service every 15 minutes.
It tests the full flow:

1. Edge device calls POST /api/intervention/suggest with user_id
2. Intervention service receives request
3. Intervention calls Fusion service internally:
   - Fusion calculates time window:
     * Gets last_fusion_timestamp = MAX(emotion_log.timestamp) for user
     * Calculates window_start = current_time - 15 minutes
     * Effective start = max(last_fusion_timestamp, window_start)
   - Fusion queries database:
     * voice_emotion: timestamp > effective_start AND timestamp <= current_time
     * face_emotion: timestamp > effective_start AND timestamp <= current_time
     * bvs_emotion: timestamp > effective_start AND timestamp <= current_time
   - Fusion runs fusion algorithm on collected signals
   - Fusion writes result to emotional_log table with current timestamp
4. Intervention reads latest emotion from emotional_log table
5. Decision engine runs (kick-start logic)
6. Suggestion engine runs (activity recommendations)
7. Results returned to edge device

IMPORTANT: This script triggers Fusion, which queries the database for signals and writes
to emotional_log. Make sure signals exist in voice_emotion, face_emotion, and/or bvs_emotion
tables before running this script, or use the simulation dashboard to generate signals.

Usage:
    python testing/test_intervention_polling.py [--user-id UUID] [--interval SECONDS] [--once] [--verbose]
"""

import os
import sys
import time
import argparse
import requests
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_USER_ID = os.getenv("DEV_USER_ID", "8517c97f-66ef-4955-86ed-531013d33d3e")
DEFAULT_BASE_URL = os.getenv("FUSION_SERVICE_URL", "http://localhost:8000")
DEFAULT_INTERVAL = 15 * 60  # 15 minutes in seconds


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_service_health(base_url: str) -> bool:
    """Check if the intervention service is accessible."""
    try:
        response = requests.get(f"{base_url}/api/intervention/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def poll_intervention_service(base_url: str, user_id: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Poll the intervention service (simulates edge device request).
    
    Args:
        base_url: Base URL of the cloud service
        user_id: User UUID
        verbose: Enable verbose logging
    
    Returns:
        Response dictionary or None on failure
    """
    endpoint = f"{base_url}/api/intervention/suggest"
    payload = {"user_id": user_id}
    
    if verbose:
        print(f"\n[Poll] Calling intervention service...")
        print(f"  URL: {endpoint}")
        print(f"  User ID: {user_id}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
    
    try:
        start_time = time.time()
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if verbose:
                print(f"  ✓ Response received in {duration:.2f}s")
            
            return result
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error details: {error_data}")
            except:
                print(f"  Error text: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection error: Could not connect to {base_url}")
        print("  Make sure the cloud service is running")
        return None
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout: Request took longer than 30s")
        return None
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def display_results(result: Dict[str, Any], verbose: bool = False):
    """Display intervention service results."""
    print_section("INTERVENTION RESULTS")
    
    # Display decision
    decision = result.get("decision", {})
    trigger = decision.get("trigger_intervention", False)
    confidence = decision.get("confidence_score", 0.0)
    reasoning = decision.get("reasoning", "N/A")
    
    print(f"\nDecision:")
    print(f"  Trigger Intervention: {'YES' if trigger else 'NO'}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Reasoning: {reasoning}")
    
    # Display suggestions
    suggestion = result.get("suggestion", {})
    activities = suggestion.get("ranked_activities", [])
    suggestion_reasoning = suggestion.get("reasoning", "N/A")
    
    print(f"\nSuggestions:")
    print(f"  Reasoning: {suggestion_reasoning}")
    print(f"  Activities ({len(activities)}):")
    
    if activities:
        for i, activity in enumerate(activities[:10], 1):  # Show top 10
            activity_type = activity.get("activity_type", "N/A")
            score = activity.get("score", 0.0)
            rank = activity.get("rank", 0)
            print(f"    {i}. {activity_type} (score: {score:.3f}, rank: {rank})")
    else:
        print("    No activities suggested")
    
    if verbose:
        print(f"\nFull Response:")
        import json
        print(json.dumps(result, indent=2))


def single_poll(base_url: str, user_id: str, verbose: bool = False):
    """Perform a single poll of the intervention service."""
    print_section("EDGE DEVICE POLLING SIMULATION")
    print(f"User ID: {user_id}")
    print(f"Service URL: {base_url}")
    print(f"Mode: Single poll")
    
    # Check service health
    print("\nChecking service health...")
    if check_service_health(base_url):
        print("  ✓ Service is healthy")
    else:
        print("  ⚠ Service health check failed, continuing anyway...")
    
    # Poll intervention service
    result = poll_intervention_service(base_url, user_id, verbose)
    
    if result:
        display_results(result, verbose)
        print("\n" + "=" * 80)
        print("✓ Poll completed successfully")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("✗ Poll failed")
        print("=" * 80)
        return False


def continuous_polling(base_url: str, user_id: str, interval: int, verbose: bool = False):
    """Continuously poll the intervention service at specified intervals."""
    print_section("EDGE DEVICE CONTINUOUS POLLING SIMULATION")
    print(f"User ID: {user_id}")
    print(f"Service URL: {base_url}")
    print(f"Poll Interval: {interval} seconds ({interval / 60:.1f} minutes)")
    print(f"Mode: Continuous (press Ctrl+C to stop)")
    
    # Check service health
    print("\nChecking service health...")
    if check_service_health(base_url):
        print("  ✓ Service is healthy")
    else:
        print("  ⚠ Service health check failed, continuing anyway...")
    
    poll_count = 0
    
    try:
        while True:
            poll_count += 1
            print(f"\n{'=' * 80}")
            print(f"Poll #{poll_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 80}")
            
            result = poll_intervention_service(base_url, user_id, verbose)
            
            if result:
                display_results(result, verbose)
            else:
                print("\n⚠ Poll failed, will retry on next interval")
            
            if poll_count < float('inf'):  # Always true, but shows intent
                print(f"\n⏳ Waiting {interval} seconds until next poll...")
                print("   (Press Ctrl+C to stop)")
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("POLLING STOPPED BY USER")
        print("=" * 80)
        print(f"Total polls completed: {poll_count}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate edge device polling intervention service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single poll
  python testing/test_intervention_polling.py --once
  
  # Continuous polling every 15 minutes
  python testing/test_intervention_polling.py
  
  # Custom interval (5 minutes)
  python testing/test_intervention_polling.py --interval 300
  
  # Verbose mode
  python testing/test_intervention_polling.py --once --verbose
        """
    )
    
    parser.add_argument(
        "--user-id",
        type=str,
        default=DEFAULT_USER_ID,
        help=f"User UUID (default: {DEFAULT_USER_ID})"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL of cloud service (default: {DEFAULT_BASE_URL})"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_INTERVAL} = 15 minutes)"
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform a single poll instead of continuous polling"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate user_id format (basic UUID check)
    if len(args.user_id) != 36 or args.user_id.count('-') != 4:
        print(f"⚠ Warning: User ID '{args.user_id}' doesn't look like a valid UUID")
        print("  Continuing anyway...")
    
    # Run polling
    if args.once:
        success = single_poll(args.base_url, args.user_id, args.verbose)
        sys.exit(0 if success else 1)
    else:
        continuous_polling(args.base_url, args.user_id, args.interval, args.verbose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(0)


