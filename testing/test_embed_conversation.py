"""
Test script for embed_conversation_messages function

This script allows you to:
1. Add test messages to a conversation
2. Embed those messages
3. Verify messages and embeddings were created

Usage:
    python testing/test_embed_conversation.py <conversation_id>
    
Or set environment variables:
    DEV_CONVERSATION_ID=<conversation_id>
    DEV_USER_ID=<user_id>
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_generator.message_preprocessor import embed_conversation_messages
from utils import database

# Load environment variables
load_dotenv()

def add_test_messages(conversation_id: str, messages: list[str]) -> bool:
    """
    Add test messages to a conversation.
    
    Args:
        conversation_id: UUID of the conversation
        messages: List of message text strings
    
    Returns:
        True if successful, False otherwise
    """
    try:
        client = database.get_supabase_client()
        
        # Insert messages
        for msg_text in messages:
            payload = {
                "conversation_id": conversation_id,
                "role": "user",
                "text": msg_text
            }
            
            response = client.table("wb_message")\
                .insert(payload)\
                .execute()
            
            if not response.data:
                print(f"Warning: Failed to insert message: {msg_text[:50]}...")
                return False
        
        print(f"✓ Successfully added {len(messages)} messages to conversation {conversation_id}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to add messages: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_messages(conversation_id: str, user_id: str = None) -> int:
    """
    Verify messages exist in the conversation.
    
    Args:
        conversation_id: UUID of the conversation
        user_id: Optional user_id for validation
    
    Returns:
        Number of user messages found
    """
    try:
        messages = database.load_conversation_messages(conversation_id, user_id=user_id)
        count = len(messages)
        print(f"✓ Found {count} user messages in conversation")
        return count
    except Exception as e:
        print(f"ERROR: Failed to verify messages: {e}")
        return 0


def verify_embeddings(conversation_id: str, user_id: str = None, model_tag: str = 'e5') -> int:
    """
    Verify embeddings exist for messages in the conversation.
    
    Args:
        conversation_id: UUID of the conversation
        user_id: Optional user_id for validation
        model_tag: Model tag to check
    
    Returns:
        Number of embeddings found
    """
    try:
        client = database.get_supabase_client()
        
        # Get all message IDs for this conversation
        messages = database.load_conversation_messages(conversation_id, user_id=user_id)
        message_ids = [msg.get("id") for msg in messages]
        
        if not message_ids:
            return 0
        
        # Check embeddings for each message (including potential chunks)
        embedding_count = 0
        for msg_id in message_ids:
            # Check for direct message embedding
            if database.check_embedding_exists(msg_id, model_tag):
                embedding_count += 1
            
            # Check for chunk embeddings (chunks use UUID v5 derived from message_id)
            # We'll check a few potential chunk indices
            import uuid
            for chunk_idx in range(5):  # Check up to 5 chunks
                try:
                    namespace = uuid.UUID(msg_id)
                    chunk_ref_id = str(uuid.uuid5(namespace, str(chunk_idx)))
                    if database.check_embedding_exists(chunk_ref_id, model_tag):
                        embedding_count += 1
                except:
                    continue
        
        print(f"✓ Found {embedding_count} embeddings for conversation (model: {model_tag})")
        return embedding_count
        
    except Exception as e:
        print(f"ERROR: Failed to verify embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 0


def get_user_id_from_conversation(conversation_id: str) -> str:
    """
    Get user_id from conversation_id.
    
    Args:
        conversation_id: UUID of the conversation
    
    Returns:
        User ID or None if not found
    """
    try:
        return database.get_conversation_user_id(conversation_id)
    except ValueError as e:
        print(f"Warning: Could not get user_id from conversation: {e}")
        return None


def main():
    """Main test function."""
    
    # Get conversation_id from command line or environment
    conversation_id = None
    if len(sys.argv) > 1:
        conversation_id = sys.argv[1]
    else:
        conversation_id = os.getenv("DEV_CONVERSATION_ID")
    
    if not conversation_id:
        print("Error: Conversation ID required")
        print("\nUsage:")
        print("  python testing/test_embed_conversation.py <conversation_id>")
        print("\nOr set environment variable:")
        print("  DEV_CONVERSATION_ID=<conversation_id>")
        sys.exit(1)
    
    # Get user_id
    user_id = os.getenv("DEV_USER_ID")
    if not user_id:
        # Try to get from conversation
        user_id = get_user_id_from_conversation(conversation_id)
        if not user_id:
            print("Error: Could not determine user_id")
            print("Please set DEV_USER_ID in your .env file")
            sys.exit(1)
    
    print("=" * 80)
    print("Test: Embed Conversation Messages")
    print("=" * 80)
    print(f"Conversation ID: {conversation_id}")
    print(f"User ID: {user_id}")
    print("=" * 80)
    
    # Step 1: Get test messages from user
    print("\nStep 1: Add test messages")
    print("-" * 80)
    print("Enter test messages (one per line, empty line to finish):")
    print("(Or press Enter to skip adding new messages and test existing ones)")
    
    test_messages = []
    while True:
        try:
            line = input("> ").strip()
            if not line:
                break
            test_messages.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    
    # Add messages if provided
    if test_messages:
        print(f"\nAdding {len(test_messages)} test messages...")
        if not add_test_messages(conversation_id, test_messages):
            print("Failed to add messages. Exiting.")
            sys.exit(1)
    else:
        print("No new messages to add. Testing with existing messages.")
    
    # Step 2: Verify messages exist
    print("\nStep 2: Verify messages")
    print("-" * 80)
    message_count = verify_messages(conversation_id, user_id=user_id)
    
    if message_count == 0:
        print("No messages found. Cannot proceed with embedding.")
        sys.exit(1)
    
    # Step 3: Embed messages
    print("\nStep 3: Embed messages")
    print("-" * 80)
    print("Embedding messages with e5 model...")
    
    try:
        result = embed_conversation_messages(conversation_id, user_id=user_id, model_tag='e5')
        
        print("\nEmbedding Results:")
        print(f"  Messages processed: {result['messages_processed']}")
        print(f"  Chunks created: {result['chunks_created']}")
        print(f"  Embeddings stored: {result['embeddings_stored']}")
        print(f"  Messages skipped: {result['messages_skipped']}")
        
        if result['embeddings_stored'] == 0 and result['messages_skipped'] > 0:
            print("\n  ℹ All messages were already embedded (idempotence working)")
        elif result['embeddings_stored'] > 0:
            print(f"\n  ✓ Successfully created {result['embeddings_stored']} new embeddings")
        
    except Exception as e:
        print(f"\nERROR: Failed to embed messages: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Verify embeddings
    print("\nStep 4: Verify embeddings")
    print("-" * 80)
    embedding_count = verify_embeddings(conversation_id, user_id=user_id, model_tag='e5')
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Messages in conversation: {message_count}")
    print(f"Embeddings found: {embedding_count}")
    
    if embedding_count > 0:
        print("\n✓ Test PASSED: Messages and embeddings are present")
    else:
        print("\n✗ Test FAILED: No embeddings found")
        print("  Check database vector column dimension (should be 768 for e5)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

