"""
Script to check what match_embeddings functions exist in the database
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import database

# Load environment variables
load_dotenv()


def check_functions():
    """Check what match_embeddings functions exist."""
    try:
        client = database.get_supabase_client()
        
        # Query pg_proc to find all match_embeddings functions
        # Note: This requires direct SQL query, which Supabase Python client doesn't support well
        # So we'll try to call the RPC and see what error we get, or use a different approach
        
        print("Checking database for match_embeddings functions...")
        print("\nNote: To see all functions, run the SQL query in Supabase SQL editor:")
        print("See: testing/check_match_embeddings_functions.sql")
        
        # Try to get function info via a test call
        # We'll use a simple approach - try calling with different parameter sets
        print("\nAttempting to identify function signatures by testing calls...")
        
        # Test 1: Without match_limit and index_limit
        print("\n1. Testing function WITHOUT match_limit/index_limit:")
        try:
            response = client.rpc(
                'match_embeddings',
                {
                    'query_vector': [0.0] * 768,  # Dummy vector
                    'match_user_id': '00000000-0000-0000-0000-000000000000',
                    'match_model_tag': 'e5',
                    'match_kind': 'message',
                    'match_threshold': 0.7
                }
            ).execute()
            print("   ✅ Function exists (without match_limit/index_limit)")
        except Exception as e:
            error_msg = str(e)
            if 'PGRST203' in error_msg or 'Multiple Choices' in error_msg:
                print("   ⚠️  Ambiguity detected - multiple functions exist")
            else:
                print(f"   ❌ Error: {error_msg[:200]}")
        
        # Test 2: With match_limit and index_limit
        print("\n2. Testing function WITH match_limit/index_limit:")
        try:
            response = client.rpc(
                'match_embeddings',
                {
                    'query_vector': [0.0] * 768,  # Dummy vector
                    'match_user_id': '00000000-0000-0000-0000-000000000000',
                    'match_model_tag': 'e5',
                    'match_kind': 'message',
                    'match_threshold': 0.7,
                    'match_limit': None,
                    'index_limit': 100
                }
            ).execute()
            print("   ✅ Function exists (with match_limit/index_limit)")
        except Exception as e:
            error_msg = str(e)
            if 'PGRST203' in error_msg or 'Multiple Choices' in error_msg:
                print("   ⚠️  Ambiguity detected - multiple functions exist")
            else:
                print(f"   ❌ Error: {error_msg[:200]}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION:")
        print("=" * 80)
        print("Run this SQL in Supabase SQL editor to see all functions:")
        print("\nSELECT p.proname, pg_get_function_arguments(p.oid) as args")
        print("FROM pg_proc p")
        print("JOIN pg_namespace n ON p.pronamespace = n.oid")
        print("WHERE n.nspname = 'public' AND p.proname = 'match_embeddings';")
        print("\nThen drop all versions and recreate with the fixed SQL.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_functions()

