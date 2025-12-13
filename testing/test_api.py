"""
Test script for the Well-Bot CMS API

This script tests the /api/context/process endpoint using DEV_USER_ID from .env file.
Run it with: python testing/test_api.py
Or from the testing directory: python test_api.py

Optional: Set DEV_CONVERSATION_ID in .env to test with conversation_id (triggers Step 0 embedding)
"""

import os
import sys
import argparse
from pathlib import Path
import requests
from dotenv import load_dotenv

# Add parent directory to path to allow imports from root (if needed)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables from parent directory
env_path = parent_dir / ".env"
load_dotenv(dotenv_path=env_path)

def test_api_endpoint(conversation_id=None):
    """Test the /api/context/process endpoint."""
    
    # Get user ID from environment variable
    user_id = os.getenv("DEV_USER_ID")
    
    if not user_id:
        print("Error: DEV_USER_ID environment variable is not set")
        print("Please set DEV_USER_ID in your .env file")
        print("\nExample:")
        print("DEV_USER_ID=96975f52-5b05-4eb1-bfa5-530485112518")
        sys.exit(1)
    
    # Get conversation_id from argument or environment variable
    if not conversation_id:
        conversation_id = os.getenv("DEV_CONVERSATION_ID")
    
    # API endpoint
    api_url = "http://localhost:8000/api/context/process"
    
    # Request payload
    payload = {
        "user_id": user_id
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    print(f"Testing API endpoint: {api_url}")
    print(f"Using user_id: {user_id}")
    if conversation_id:
        print(f"Using conversation_id: {conversation_id} (will trigger Step 0: embedding)")
    else:
        print("No conversation_id provided (skipping Step 0: embedding)")
    print("\n" + "=" * 60)
    print("Sending request...")
    print("=" * 60)
    if conversation_id:
        print("\nNote: This may take 2-6 minutes (embedding + both LLM extractions)")
    else:
        print("\nNote: This may take 2-6 minutes (both LLM extractions)")
    print("Processing...\n")
    
    try:
        # Make POST request
        response = requests.post(api_url, json=payload, timeout=600)  # 10 minute timeout
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        print("\n" + "=" * 60)
        print("SUCCESS! Response received:")
        print("=" * 60)
        print(f"\nStatus: {result.get('status')}")
        print(f"User ID: {result.get('user_id')}")
        
        # Display facts if available
        if result.get('facts'):
            print("\n" + "-" * 60)
            print("PERSONA FACTS:")
            print("-" * 60)
            print(result.get('facts'))
        else:
            print("\nFacts: Not extracted (may have failed)")
        
        # Display persona summary if available
        if result.get('persona_summary'):
            print("\n" + "-" * 60)
            print("DAILY LIFE CONTEXT:")
            print("-" * 60)
            print(result.get('persona_summary'))
        else:
            print("\nPersona Summary: Not extracted")
        
        print("\n" + "=" * 60)
        print("Both results have been saved to the database!")
        print("=" * 60)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the API server")
        print("Make sure the server is running from the root directory: python main.py")
        return False
    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out (took longer than 10 minutes)")
        print("The server may still be processing. Check server logs.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: HTTP {e.response.status_code}")
        try:
            error_detail = e.response.json()
            print(f"Detail: {error_detail.get('detail', 'Unknown error')}")
        except:
            print(f"Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Well-Bot CMS API context processing endpoint")
    parser.add_argument(
        "--conversation-id",
        type=str,
        help="Optional conversation ID to test with (triggers Step 0: embedding new messages)"
    )
    args = parser.parse_args()
    
    success = test_api_endpoint(conversation_id=args.conversation_id)
    sys.exit(0 if success else 1)

