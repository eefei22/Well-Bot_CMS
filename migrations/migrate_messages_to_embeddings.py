"""
Migration Script: Embed All Historical Messages

This script performs a one-time migration to embed all historical user messages
and store them in the wb_embeddings table.

Usage:
    python migrations/migrate_messages_to_embeddings.py [--model-tag e5] [--user-id USER_ID] [--resume]
    
Options:
    --model-tag: Embedding model to use ('miniLM' or 'e5'), default 'e5'
    --user-id: Process only specific user (for testing)
    --resume: Resume from last processed user (if progress file exists)
"""

import os
import sys
import argparse
import logging
import uuid
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import database
from context_generator.message_preprocessor import (
    _filter_messages,
    _normalize_message,
    _chunk_message
)
from utils.embeddings import generate_embedding

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Progress file for resumability
PROGRESS_FILE = "migrations/migration_progress.txt"


def load_progress() -> List[str]:
    """
    Load list of already processed user IDs from progress file.
    
    Returns:
        List of user IDs that have been processed
    """
    if not os.path.exists(PROGRESS_FILE):
        return []
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.warning(f"Failed to load progress file: {e}")
        return []


def save_progress(user_id: str):
    """
    Save processed user ID to progress file.
    
    Args:
        user_id: User ID that was processed
    """
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, 'a') as f:
            f.write(f"{user_id}\n")
    except Exception as e:
        logger.warning(f"Failed to save progress for user {user_id}: {e}")


def migrate_user_messages(user_id: str, model_tag: str = 'e5') -> Dict:
    """
    Migrate all messages for a single user.
    
    Args:
        user_id: UUID of the user
        model_tag: Model tag for embeddings ('miniLM' or 'e5')
    
    Returns:
        Dictionary with migration statistics
    """
    logger.info(f"Starting migration for user {user_id} with model {model_tag}")
    
    # Load all conversations for this user
    conversations = database.load_user_messages(user_id)
    
    if not conversations:
        logger.warning(f"No conversations found for user {user_id}")
        return {
            "user_id": user_id,
            "conversations_processed": 0,
            "messages_processed": 0,
            "chunks_created": 0,
            "embeddings_stored": 0,
            "messages_skipped": 0,
            "errors": 0
        }
    
    total_messages_processed = 0
    total_chunks_created = 0
    total_embeddings_stored = 0
    total_messages_skipped = 0
    total_errors = 0
    
    # Process each conversation
    for conv in conversations:
        conversation_id = conv.get("conversation_id")
        
        # Load messages with IDs from database
        conv_messages = database.load_conversation_messages(conversation_id)
        
        if not conv_messages:
            continue
        
        logger.info(f"Processing conversation {conversation_id} ({len(conv_messages)} messages)")
        
        # Extract message texts and IDs, then filter
        message_data = []
        for msg in conv_messages:
            msg_text = msg.get("text", "")
            if msg_text and _filter_messages([msg_text], min_words=4):
                message_data.append((msg.get("id"), msg_text))
        
        if not message_data:
            continue
        
        # Process each message
        for message_id, message_text in message_data:
            try:
                # Normalize message
                normalized_text = _normalize_message(message_text)
                
                # Chunk message if needed
                chunks = _chunk_message(normalized_text, threshold=500)
                total_chunks_created += len(chunks)
                
                # Process each chunk
                for chunk_text, chunk_index in chunks:
                    # Generate ref_id (same logic as embed_conversation_messages)
                    if len(chunks) > 1:
                        # Create deterministic UUID v5 from message_id (as namespace) + chunk_index
                        namespace = uuid.UUID(message_id)
                        ref_id = str(uuid.uuid5(namespace, str(chunk_index)))
                    else:
                        ref_id = message_id
                    
                    # Check idempotence
                    if database.check_embedding_exists(ref_id, model_tag):
                        total_messages_skipped += 1
                        continue
                    
                    # Generate and store embedding
                    try:
                        vector = generate_embedding(chunk_text, model_tag=model_tag)
                        
                        success = database.store_embedding(
                            user_id=user_id,
                            kind="message",
                            ref_id=ref_id,
                            vector=vector,
                            model_tag=model_tag
                        )
                        
                        if success:
                            total_embeddings_stored += 1
                        else:
                            total_errors += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to embed chunk for message {message_id}, chunk {chunk_index}: {e}")
                        total_errors += 1
                        continue
                
                total_messages_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing message {message_id} in conversation {conversation_id}: {e}")
                total_errors += 1
                continue
    
    logger.info(
        f"Completed migration for user {user_id}: "
        f"{total_messages_processed} messages, {total_chunks_created} chunks, "
        f"{total_embeddings_stored} embeddings stored, {total_messages_skipped} skipped, {total_errors} errors"
    )
    
    return {
        "user_id": user_id,
        "conversations_processed": len(conversations),
        "messages_processed": total_messages_processed,
        "chunks_created": total_chunks_created,
        "embeddings_stored": total_embeddings_stored,
        "messages_skipped": total_messages_skipped,
        "errors": total_errors
    }


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate all historical messages to embeddings")
    parser.add_argument(
        "--model-tag",
        type=str,
        default="e5",
        choices=["miniLM", "e5"],
        help="Embedding model to use (default: e5)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Process only specific user (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last processed user"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Starting migration of historical messages to embeddings")
    logger.info(f"Model tag: {args.model_tag}")
    logger.info("=" * 80)
    
    # Get users to process
    if args.user_id:
        user_ids = [args.user_id]
        logger.info(f"Processing single user: {args.user_id}")
    else:
        user_ids = database.get_all_users()
        logger.info(f"Found {len(user_ids)} users to process")
        
        # Filter out already processed users if resuming
        if args.resume:
            processed_users = load_progress()
            user_ids = [uid for uid in user_ids if uid not in processed_users]
            logger.info(f"Resuming: {len(processed_users)} already processed, {len(user_ids)} remaining")
    
    if not user_ids:
        logger.info("No users to process")
        return
    
    # Process each user
    total_stats = {
        "users_processed": 0,
        "total_messages": 0,
        "total_chunks": 0,
        "total_embeddings": 0,
        "total_skipped": 0,
        "total_errors": 0
    }
    
    for i, user_id in enumerate(user_ids, 1):
        logger.info(f"\n[{i}/{len(user_ids)}] Processing user {user_id}")
        
        try:
            stats = migrate_user_messages(user_id, model_tag=args.model_tag)
            
            total_stats["users_processed"] += 1
            total_stats["total_messages"] += stats["messages_processed"]
            total_stats["total_chunks"] += stats["chunks_created"]
            total_stats["total_embeddings"] += stats["embeddings_stored"]
            total_stats["total_skipped"] += stats["messages_skipped"]
            total_stats["total_errors"] += stats["errors"]
            
            # Save progress
            save_progress(user_id)
            
        except Exception as e:
            logger.error(f"Failed to process user {user_id}: {e}")
            total_stats["total_errors"] += 1
            continue
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("Migration Summary")
    logger.info("=" * 80)
    logger.info(f"Users processed: {total_stats['users_processed']}/{len(user_ids)}")
    logger.info(f"Total messages processed: {total_stats['total_messages']}")
    logger.info(f"Total chunks created: {total_stats['total_chunks']}")
    logger.info(f"Total embeddings stored: {total_stats['total_embeddings']}")
    logger.info(f"Total messages skipped (already embedded): {total_stats['total_skipped']}")
    logger.info(f"Total errors: {total_stats['total_errors']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

