"""
Database Script

This script handles Supabase database connection and read operations.
Combines database connection and message loading functionality.
"""

import os
from typing import Dict, List
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supabase_config() -> Dict[str, str]:
    """
    Get Supabase configuration from environment variables.
    
    Returns:
        Dictionary with 'url' and 'service_role_key'
    
    Raises:
        ValueError: If required environment variables are missing
    """
    url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not service_role_key:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")
    
    return {
        "url": url,
        "service_role_key": service_role_key
    }


def get_supabase_client(service: bool = True) -> Client:
    """
    Create and return a Supabase client instance.
    
    Args:
        service: If True, use service_role_key (for admin operations).
                If False, use anon_key (for user-scoped operations).
    
    Returns:
        Supabase Client instance
    """
    config = get_supabase_config()
    url = config["url"]
    key = config["service_role_key"] if service else os.getenv("SUPABASE_ANON_KEY", "")
    
    if not key:
        raise ValueError("Service role key or anon key is required")
    
    client = create_client(url, key)
    logger.info("Successfully connected to Supabase")
    return client


def load_user_messages(user_id: str) -> List[Dict]:
    """
    Load all user messages from public.wb_message for a specific user, grouped by conversation.
    
    This function can be called by context_processor.py to retrieve user messages.
    
    Args:
        user_id: UUID of the user
    
    Returns:
        List of conversation dictionaries, each containing:
        - user_id: UUID of the user
        - conversation_id: UUID of the conversation
        - conversation_created_at: Timestamp when conversation started
        - total_messages: Count of user messages in this conversation
        - messages: Array of user messages (filtered to role="user" only)
          Each message contains: {"text": "..."}
    """
    client = get_supabase_client()
    
    # Get all conversations for this user with their metadata
    conversations_response = client.table("wb_conversation")\
        .select("id, started_at")\
        .eq("user_id", user_id)\
        .order("started_at", desc=False)\
        .execute()
    
    conversations = conversations_response.data
    
    if not conversations:
        logger.info(f"No conversations found for user {user_id}")
        return []
    
    logger.info(f"Found {len(conversations)} conversations for user {user_id}")
    
    # Process each conversation
    result = []
    for conv in conversations:
        conv_id = conv['id']
        conv_started_at = conv['started_at']
        
        # Get all messages for this conversation (only user role)
        messages_response = client.table("wb_message")\
            .select("text")\
            .eq("conversation_id", conv_id)\
            .eq("role", "user")\
            .order("created_at", desc=False)\
            .execute()
        
        user_messages = messages_response.data
        
        # Format messages to only include text field
        formatted_messages = [{"text": msg["text"]} for msg in user_messages]
        
        # Only include conversations that have user messages
        if formatted_messages:
            conversation_data = {
                "user_id": user_id,
                "conversation_id": conv_id,
                "conversation_created_at": conv_started_at,
                "total_messages": len(formatted_messages),
                "messages": formatted_messages
            }
            result.append(conversation_data)
            logger.debug(f"Conversation {conv_id}: {len(formatted_messages)} user messages")
    
    total_user_messages = sum(conv['total_messages'] for conv in result)
    logger.info(f"Loaded {total_user_messages} user messages from {len(result)} conversations")
    return result


def write_user_context_bundle(user_id: str, persona_summary: str = None, facts: str = None) -> bool:
    """
    Write or update a user context bundle in the user_context_bundle table.
    This uses upsert (insert or update) since user_id is the primary key.
    Supports partial updates - can update one or both fields.
    
    Args:
        user_id: UUID of the user
        persona_summary: Optional persona summary text to save (daily life context)
        facts: Optional persona facts text to save
    
    Returns:
        True if write/update succeeded, False otherwise
    """
    try:
        client = get_supabase_client()
        payload = {
            "user_id": user_id
        }
        
        # Only include fields that are provided
        if persona_summary is not None:
            payload["persona_summary"] = persona_summary
        if facts is not None:
            payload["facts"] = facts
        
        # Use upsert to insert or update (since user_id is primary key)
        response = client.table("user_context_bundle")\
            .upsert(payload, on_conflict="user_id")\
            .execute()
        
        if response.data:
            logger.info(f"Successfully wrote/updated context bundle for user {user_id}")
            return True
        else:
            logger.warning(f"No data returned for context bundle update for user {user_id}")
            return False
    except Exception as e:
        logger.error(f"Failed to write context bundle for user {user_id}: {e}")
        return False
