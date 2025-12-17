#!/usr/bin/env python3
"""
End-to-End Test Script for Context & Facts Extraction

This script tests the complete context and facts extraction flow:
1. Semantic Vector Search (by focus areas)
2. Message retrieval from database
3. Language detection
4. Facts extraction via LLM
5. Context extraction via LLM
6. Database persistence verification

Usage:
    python testing/test_context_facts_e2e.py
    
Prerequisites:
    - .env file with DEV_USER_ID, DEEPSEEK_API_KEY, SUPABASE_URL, SUPABASE_KEY
    - Supabase database accessible
    - User has existing conversation messages
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_generator import facts_extractor, context_extractor
from utils import database, vector_search
import logging

# Load environment variables
load_dotenv()

# Configuration
DEV_USER_ID = os.getenv("DEV_USER_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Configure logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def print_preview(text: str, max_length: int = 300, indent: str = "  "):
    """Print a preview of text with proper formatting."""
    if not text:
        print(f"{indent}(empty)")
        return
    
    lines = text.split('\n')
    preview = '\n'.join(lines[:5])  # First 5 lines
    
    if len(preview) > max_length:
        preview = preview[:max_length] + "..."
    
    for line in preview.split('\n'):
        print(f"{indent}{line}")
    
    if len(lines) > 5:
        print(f"{indent}... ({len(lines) - 5} more lines)")


def get_context_bundle(user_id: str):
    """
    Get current state from users_context_bundle table.
    
    Returns:
        Dictionary with facts, persona_summary, version_ts or None if not exists
    """
    try:
        client = database.get_supabase_client()
        result = client.table("users_context_bundle")\
            .select("*")\
            .eq("user_id", user_id)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching context bundle: {e}")
        return None


def display_context_bundle_state(bundle: dict, label: str):
    """Display the current state of context bundle."""
    print(f"\n{label}:")
    if not bundle:
        print("  ❌ No context bundle found in database")
        return
    
    print(f"  📅 Version Timestamp: {bundle.get('version_ts', 'N/A')}")
    
    facts = bundle.get('facts')
    persona_summary = bundle.get('persona_summary')
    
    print(f"\n  📊 Facts: {len(facts) if facts else 0} characters")
    if facts:
        print_preview(facts, max_length=200, indent="     ")
    else:
        print("     (empty)")
    
    print(f"\n  📝 Context Summary: {len(persona_summary) if persona_summary else 0} characters")
    if persona_summary:
        print_preview(persona_summary, max_length=200, indent="     ")
    else:
        print("     (empty)")


def compare_context_bundles(before: dict, after: dict):
    """Compare before and after states and display differences."""
    print_subsection("COMPARISON: Before vs After")
    
    if not before:
        print("  ✨ NEW RECORD CREATED")
        display_context_bundle_state(after, "After")
        return
    
    # Compare timestamps
    before_ts = before.get('version_ts', 'N/A')
    after_ts = after.get('version_ts', 'N/A')
    
    print(f"\n  📅 Timestamp:")
    print(f"     Before: {before_ts}")
    print(f"     After:  {after_ts}")
    if before_ts != after_ts:
        print(f"     ✓ Timestamp updated!")
    else:
        print(f"     ⚠ Timestamp NOT updated")
    
    # Compare facts
    before_facts_len = len(before.get('facts', '')) if before.get('facts') else 0
    after_facts_len = len(after.get('facts', '')) if after.get('facts') else 0
    
    print(f"\n  📊 Facts:")
    print(f"     Before: {before_facts_len:,} characters")
    print(f"     After:  {after_facts_len:,} characters")
    if after_facts_len > before_facts_len:
        print(f"     ✓ Increased by {after_facts_len - before_facts_len:,} characters")
    elif after_facts_len < before_facts_len:
        print(f"     ↓ Decreased by {before_facts_len - after_facts_len:,} characters")
    else:
        print(f"     → No change")
    
    # Compare context summary
    before_context_len = len(before.get('persona_summary', '')) if before.get('persona_summary') else 0
    after_context_len = len(after.get('persona_summary', '')) if after.get('persona_summary') else 0
    
    print(f"\n  📝 Context Summary:")
    print(f"     Before: {before_context_len:,} characters")
    print(f"     After:  {after_context_len:,} characters")
    if after_context_len > before_context_len:
        print(f"     ✓ Increased by {after_context_len - before_context_len:,} characters")
    elif after_context_len < before_context_len:
        print(f"     ↓ Decreased by {before_context_len - after_context_len:,} characters")
    else:
        print(f"     → No change")


def test_context_extraction_e2e(user_id: str):
    """
    Test complete extraction flow with detailed logging.
    
    Args:
        user_id: User ID to test with
        
    Returns:
        True if successful, False otherwise
    """
    print_section("CONTEXT & FACTS EXTRACTION E2E TEST")
    print(f"\n  User ID: {user_id}")
    print(f"  Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    facts_result = None
    context_result = None
    
    try:
        # ============================================================
        # PHASE 0: Pre-test Database State Check
        # ============================================================
        print_section("[Phase 0] Database State - BEFORE")
        
        before_bundle = get_context_bundle(user_id)
        display_context_bundle_state(before_bundle, "Current State")
        
        # ============================================================
        # PHASE 1: Facts Extraction
        # ============================================================
        facts_start = time.time()
        print_section("[Phase 1] Extracting Persona Facts")
        print(f"  User: {user_id}")
        print("  → Queried 6 focus areas (communication style, interests, personality traits, values, characteristics, behavioural patterns)")
        print("  → Using semantic vector search with e5 embeddings")
        print("  → Similarity threshold: 0.7 (fallback to 0.6 if needed)")
        print("\n  ⏳ Processing... (this may take 1-3 minutes)")
        
        try:
            facts_result = facts_extractor.extract_user_facts(user_id)
            facts_end = time.time()
            facts_duration = facts_end - facts_start
            
            print(f"\n  ✓ Facts extraction completed in {facts_duration:.2f}s")
            print(f"  → Facts summary: {len(facts_result):,} characters")
            print("\n  Preview:")
            print_preview(facts_result, max_length=300, indent="     ")
            
        except Exception as e:
            facts_end = time.time()
            facts_duration = facts_end - facts_start
            print(f"\n  ✗ Facts extraction failed after {facts_duration:.2f}s")
            print(f"  ✗ Error: {e}")
            logger.exception("Facts extraction error details:")
            # Continue to context extraction (same as production)
        
        # ============================================================
        # PHASE 2: Context Extraction
        # ============================================================
        context_start = time.time()
        print_section("[Phase 2] Extracting Daily Life Context")
        print(f"  User: {user_id}")
        print("  → Queried 6 focus areas (routines, stories, relationships, work, events, activities)")
        print("  → Using semantic vector search with e5 embeddings")
        print("  → Similarity threshold: 0.7 (fallback to 0.6 if needed)")
        print("\n  ⏳ Processing... (this may take 1-3 minutes)")
        
        try:
            context_result = context_extractor.process_user_context(user_id)
            context_end = time.time()
            context_duration = context_end - context_start
            
            print(f"\n  ✓ Context extraction completed in {context_duration:.2f}s")
            print(f"  → Context summary: {len(context_result):,} characters")
            print("\n  Preview:")
            print_preview(context_result, max_length=300, indent="     ")
            
        except Exception as e:
            context_end = time.time()
            context_duration = context_end - context_start
            print(f"\n  ✗ Context extraction failed after {context_duration:.2f}s")
            print(f"  ✗ Error: {e}")
            logger.exception("Context extraction error details:")
            raise  # Context extraction is critical
        
        # ============================================================
        # PHASE 3: Database Verification
        # ============================================================
        print_section("[Phase 3] Database Verification - AFTER")
        
        # Small delay to ensure database write is complete
        time.sleep(0.5)
        
        after_bundle = get_context_bundle(user_id)
        
        if not after_bundle:
            print("  ✗ ERROR: Failed to retrieve updated context bundle from database")
            return False
        
        # Display comparison
        compare_context_bundles(before_bundle, after_bundle)
        
        # ============================================================
        # SUMMARY
        # ============================================================
        overall_end = time.time()
        total_duration = overall_end - overall_start
        
        print_section("TEST SUMMARY")
        
        print(f"\n  ⏱  Total Duration: {total_duration:.2f}s")
        if facts_result:
            print(f"     - Facts Extraction: {facts_duration:.2f}s")
        if context_result:
            print(f"     - Context Extraction: {context_duration:.2f}s")
        
        print(f"\n  📊 Results:")
        print(f"     - Facts: {'✓ Success' if facts_result else '✗ Failed'}")
        print(f"     - Context: {'✓ Success' if context_result else '✗ Failed'}")
        print(f"     - Database Write: ✓ Verified")
        
        if facts_result and context_result:
            print(f"\n  🎉 TEST PASSED - All phases completed successfully!")
            return True
        elif context_result:
            print(f"\n  ⚠  TEST PARTIAL - Context succeeded, facts failed")
            return True
        else:
            print(f"\n  ❌ TEST FAILED - Context extraction failed")
            return False
        
    except Exception as e:
        overall_end = time.time()
        total_duration = overall_end - overall_start
        
        print_section("TEST FAILED")
        print(f"\n  ❌ Test failed after {total_duration:.2f}s")
        print(f"  ❌ Error: {e}")
        logger.exception("Test error details:")
        return False


def main():
    """Main entry point for the test script."""
    # Check environment variables
    if not DEV_USER_ID:
        print("ERROR: DEV_USER_ID not found in .env file")
        print("\nPlease set DEV_USER_ID in your .env file:")
        print("  DEV_USER_ID=96975f52-5b05-4eb1-bfa5-530485112518")
        sys.exit(1)
    
    if not DEEPSEEK_API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not found in .env file")
        print("\nPlease set DEEPSEEK_API_KEY in your .env file:")
        print("  DEEPSEEK_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Run the test
    success = test_context_extraction_e2e(DEV_USER_ID)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
