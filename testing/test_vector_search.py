"""
Test script for vector search semantic similarity functionality

This script allows you to test semantic similarity search by:
1. Entering a query text
2. Seeing which messages match (with similarity scores)
3. Viewing the actual message texts

Usage:
    python testing/test_vector_search.py <user_id> [query_text]
    
Or set environment variables:
    DEV_USER_ID=<user_id>
    DEV_QUERY=<query_text>
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import vector_search, database

# Load environment variables
load_dotenv()


def test_semantic_search(user_id: str, query_text: str, model_tag: str = 'e5', similarity_threshold: float = 0.7):
    """
    Test semantic similarity search with a query.
    
    Args:
        user_id: UUID of the user
        query_text: Query text to search for
        model_tag: Model tag ('miniLM' or 'e5'), default 'e5'
        similarity_threshold: Minimum similarity score, default 0.7
    """
    print("\n" + "=" * 80)
    print("SEMANTIC SIMILARITY SEARCH TEST")
    print("=" * 80)
    print(f"User ID: {user_id}")
    print(f"Query: '{query_text}'")
    print(f"Model: {model_tag}")
    print(f"Similarity Threshold: {similarity_threshold}")
    print("=" * 80 + "\n")
    
    try:
        # Perform semantic search
        print("Performing semantic search...")
        results = vector_search.query_embeddings_by_semantic_prompt(
            user_id=user_id,
            query_text=query_text,
            model_tag=model_tag,
            similarity_threshold=similarity_threshold,
            kind='message'
        )
        
        if not results:
            print(f"\n❌ No results found above threshold {similarity_threshold}")
            print("\nTry:")
            print("  - Lowering the similarity threshold (e.g., 0.6 or 0.5)")
            print("  - Using a different query")
            print("  - Checking if embeddings exist for this user")
            return
        
        print(f"\n✅ Found {len(results)} matching messages\n")
        
        # Extract ref_ids
        ref_ids = [str(result['ref_id']) for result in results]
        
        # Retrieve message texts
        print("Retrieving message texts...")
        message_texts = vector_search.retrieve_message_texts(ref_ids)
        
        # Display results
        print("\n" + "-" * 80)
        print("SEARCH RESULTS")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            ref_id = result['ref_id']
            similarity = result['similarity_score']
            created_at = result.get('created_at', 'N/A')
            
            # Get message text
            msg_text = message_texts.get(str(ref_id), "[Message text not found]")
            
            # Truncate long messages for display
            display_text = msg_text
            if len(display_text) > 150:
                display_text = display_text[:147] + "..."
            
            print(f"\n[{i}] Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
            print(f"    Ref ID: {ref_id}")
            print(f"    Created: {created_at}")
            print(f"    Message: {display_text}")
        
        print("\n" + "-" * 80)
        print(f"\nTotal: {len(results)} messages matched")
        print(f"Average similarity: {sum(r['similarity_score'] for r in results) / len(results):.4f}")
        print(f"Highest similarity: {max(r['similarity_score'] for r in results):.4f}")
        print(f"Lowest similarity: {min(r['similarity_score'] for r in results):.4f}")
        
        # Option to view full message texts
        print("\n" + "-" * 80)
        view_full = input("\nView full message texts? (y/n): ").strip().lower()
        if view_full == 'y':
            print("\n" + "=" * 80)
            print("FULL MESSAGE TEXTS")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                ref_id = result['ref_id']
                similarity = result['similarity_score']
                msg_text = message_texts.get(str(ref_id), "[Message text not found]")
                print(f"\n[{i}] Similarity: {similarity:.4f}")
                print(f"Ref ID: {ref_id}")
                print(f"Message:\n{msg_text}")
                print("-" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during semantic search: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_multiple_queries(user_id: str, model_tag: str = 'e5'):
    """
    Test multiple predefined queries to see how different focus areas perform.
    
    Args:
        user_id: UUID of the user
        model_tag: Model tag ('miniLM' or 'e5'), default 'e5'
    """
    focus_areas = [
        "daily routines and activities",
        "stories and experiences the user shares",
        "people they meet and their relationships",
        "work life and professional context",
        "life events and significant moments",
        "day-to-day activities and interactions"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE FOCUS AREAS")
    print("=" * 80)
    print(f"User ID: {user_id}")
    print(f"Model: {model_tag}")
    print("=" * 80 + "\n")
    
    for i, query in enumerate(focus_areas, 1):
        print(f"\n[{i}/{len(focus_areas)}] Testing: '{query}'")
        print("-" * 80)
        
        try:
            results = vector_search.query_embeddings_by_semantic_prompt(
                user_id=user_id,
                query_text=query,
                model_tag=model_tag,
                similarity_threshold=0.7,
                kind='message'
            )
            
            if results:
                print(f"✅ Found {len(results)} results")
                print(f"   Similarity range: {min(r['similarity_score'] for r in results):.4f} - {max(r['similarity_score'] for r in results):.4f}")
            else:
                print(f"❌ No results found")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to run the test."""
    # Get user ID
    user_id = None
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = os.getenv("DEV_USER_ID")
    
    if not user_id:
        print("Error: User ID is required")
        print("\nUsage:")
        print("  python testing/test_vector_search.py <user_id> [query_text]")
        print("\nOr set environment variable:")
        print("  DEV_USER_ID=<user_id>")
        print("  DEV_QUERY=<query_text>")
        sys.exit(1)
    
    # Get query text
    query_text = None
    if len(sys.argv) > 2:
        query_text = " ".join(sys.argv[2:])
    else:
        query_text = os.getenv("DEV_QUERY")
    
    # Interactive mode
    if not query_text:
        print("\n" + "=" * 80)
        print("VECTOR SEARCH TEST SCRIPT")
        print("=" * 80)
        print(f"User ID: {user_id}")
        print("\nOptions:")
        print("  1. Enter a custom query")
        print("  2. Test all focus areas")
        print("  3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            query_text = input("\nEnter your query: ").strip()
            if not query_text:
                print("Error: Query cannot be empty")
                sys.exit(1)
            
            # Get threshold
            threshold_input = input("Similarity threshold (default 0.7): ").strip()
            threshold = float(threshold_input) if threshold_input else 0.7
            
            # Get model tag
            model_input = input("Model tag (e5/miniLM, default e5): ").strip()
            model_tag = model_input if model_input in ['e5', 'miniLM'] else 'e5'
            
            test_semantic_search(user_id, query_text, model_tag, threshold)
            
        elif choice == '2':
            model_input = input("Model tag (e5/miniLM, default e5): ").strip()
            model_tag = model_input if model_input in ['e5', 'miniLM'] else 'e5'
            test_multiple_queries(user_id, model_tag)
            
        else:
            print("Exiting...")
            sys.exit(0)
    else:
        # Non-interactive mode with provided query
        test_semantic_search(user_id, query_text)


if __name__ == "__main__":
    main()

