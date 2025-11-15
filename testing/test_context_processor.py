"""
Test script for context_processor.py

This script tests the process_user_context function.
Run it with: python test_context_processor.py
"""

import os
import sys
from dotenv import load_dotenv
from context_generator import context_processor

# Load environment variables
load_dotenv()

def test_process_user_context():
    """Test the process_user_context function."""
    
    # Get user ID from environment variable or use default test user
    user_id = os.getenv("DEV_USER_ID")
    
    if not user_id:
        print("Error: DEV_USER_ID environment variable is not set")
        print("Please set DEV_USER_ID in your .env file")
        print("\nExample:")
        print("DEV_USER_ID=8517c97f-66ef-4955-86ed-531013d33d3e")
        sys.exit(1)
    
    # Check if API key is set
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable is not set")
        print("Please set DEEPSEEK_API_KEY in your .env file")
        sys.exit(1)
    
    print(f"Testing context processor for user: {user_id}\n")
    print("=" * 60)
    
    try:
        # Process user context
        persona_summary = context_processor.process_user_context(user_id)
        
        print("\n" + "=" * 60)
        print("SUCCESS! Persona summary generated:")
        print("=" * 60)
        print(persona_summary)
        print("=" * 60)
        print(f"\nPersona saved to database for user {user_id}")
        
        return True
        
    except ValueError as e:
        print(f"\nERROR: {e}")
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_process_user_context()
    sys.exit(0 if success else 1)



