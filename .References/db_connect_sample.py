"""
Standalone Supabase Database Connection Sample Script

This script demonstrates how to:
- Connect to Supabase database
- Read from wb_conversation and wb_message tables
- Write to context_fact and user_context_bundle tables
- Work with specific users identified by UUID

Requirements:
- Set environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
- Install: pip install supabase python-dotenv

Usage:
    from db_connect_sample import (
        get_supabase_client,
        read_user_conversations,
        read_conversation_messages,
        write_context_fact,
        write_user_context_bundle
    )
    
    client = get_supabase_client()
    user_id = "8517c97f-66ef-4955-86ed-531013d33d3e"
    
    # Read conversations
    conversations = read_user_conversations(client, user_id)
    
    # Read messages for a conversation
    if conversations:
        messages = read_conversation_messages(client, conversations[0]['id'])
    
    # Write context fact
    fact_id = write_context_fact(
        client, user_id, 
        text="User prefers morning meditation",
        tags=["preference", "meditation"],
        confidence=0.85
    )
    
    # Write user context bundle
    write_user_context_bundle(
        client, user_id,
        persona_summary="Active user interested in mindfulness",
        last_session_summary="Discussed stress management",
        facts=[{"fact_id": fact_id, "text": "User prefers morning meditation"}]
    )
"""

import os
from typing import Optional, Dict, Any, List
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


def read_user_conversations(
    client: Client, 
    user_id: str, 
    limit: int = 20,
    include_ended: bool = True
) -> List[Dict[str, Any]]:
    """
    Read conversations for a specific user from wb_conversation table.
    
    Args:
        client: Supabase client instance
        user_id: UUID of the user
        limit: Maximum number of conversations to return
        include_ended: If False, only return active conversations (ended_at is NULL)
    
    Returns:
        List of conversation dictionaries with keys: id, user_id, started_at, ended_at, reason_ended
    """
    try:
        query = client.table("wb_conversation")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("started_at", desc=True)\
            .limit(limit)
        
        if not include_ended:
            query = query.is_("ended_at", "null")
        
        response = query.execute()
        
        logger.info(f"Found {len(response.data)} conversations for user {user_id}")
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to read conversations for user {user_id}: {e}")
        return []


def read_conversation_messages(
    client: Client,
    conversation_id: str,
    limit: int = 100,
    order_asc: bool = True
) -> List[Dict[str, Any]]:
    """
    Read messages for a specific conversation from wb_message table.
    
    Args:
        client: Supabase client instance
        conversation_id: UUID of the conversation
        limit: Maximum number of messages to return
        order_asc: If True, order by created_at ascending (oldest first).
                   If False, order by created_at descending (newest first).
    
    Returns:
        List of message dictionaries with keys: id, conversation_id, role, text, 
        created_at, tokens, metadata
    """
    try:
        query = client.table("wb_message")\
            .select("*")\
            .eq("conversation_id", conversation_id)\
            .order("created_at", desc=not order_asc)\
            .limit(limit)
        
        response = query.execute()
        
        logger.info(f"Found {len(response.data)} messages for conversation {conversation_id}")
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to read messages for conversation {conversation_id}: {e}")
        return []


def read_user_messages(
    client: Client,
    user_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Read all messages for a user across all their conversations.
    This joins wb_message with wb_conversation to filter by user_id.
    
    Args:
        client: Supabase client instance
        user_id: UUID of the user
        limit: Maximum number of messages to return
    
    Returns:
        List of message dictionaries with conversation context
    """
    try:
        # First get conversation IDs for this user
        conversations = read_user_conversations(client, user_id, limit=1000)
        conversation_ids = [conv['id'] for conv in conversations]
        
        if not conversation_ids:
            logger.info(f"No conversations found for user {user_id}")
            return []
        
        # Get messages from all conversations
        all_messages = []
        for conv_id in conversation_ids[:50]:  # Limit to avoid too many queries
            messages = read_conversation_messages(client, conv_id, limit=limit)
            all_messages.extend(messages)
        
        # Sort by created_at and limit
        all_messages.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        result = all_messages[:limit]
        
        logger.info(f"Found {len(result)} messages for user {user_id}")
        return result
    except Exception as e:
        logger.error(f"Failed to read messages for user {user_id}: {e}")
        return []


def write_context_fact(
    client: Client,
    user_id: str,
    text: str,
    tags: Optional[List[str]] = None,
    confidence: float = 0.0,
    recency_days: float = 0.0
) -> Optional[int]:
    """
    Write a context fact to the context_fact table.
    
    Args:
        client: Supabase client instance
        user_id: UUID of the user this fact belongs to
        text: The fact text content
        tags: Optional list of tag strings
        confidence: Confidence score (0.0 to 1.0)
        recency_days: Number of days since this fact was relevant
    
    Returns:
        The fact_id of the inserted record, or None if insertion failed
    """
    try:
        payload = {
            "user_id": user_id,
            "text": text,
            "tags": tags or [],
            "confidence": confidence,
            "recency_days": recency_days
        }
        
        response = client.table("context_fact").insert(payload).execute()
        
        if response.data and len(response.data) > 0:
            fact_id = response.data[0]["fact_id"]
            logger.info(f"Successfully created context fact {fact_id} for user {user_id}")
            return fact_id
        else:
            logger.warning("Insert succeeded but no data returned")
            return None
    except Exception as e:
        logger.error(f"Failed to write context fact for user {user_id}: {e}")
        return None


def update_context_fact(
    client: Client,
    fact_id: int,
    text: Optional[str] = None,
    tags: Optional[List[str]] = None,
    confidence: Optional[float] = None,
    recency_days: Optional[float] = None
) -> bool:
    """
    Update an existing context fact.
    
    Args:
        client: Supabase client instance
        fact_id: The fact_id to update
        text: Optional new text content
        tags: Optional new tags list
        confidence: Optional new confidence score
        recency_days: Optional new recency_days value
    
    Returns:
        True if update succeeded, False otherwise
    """
    try:
        payload = {}
        if text is not None:
            payload["text"] = text
        if tags is not None:
            payload["tags"] = tags
        if confidence is not None:
            payload["confidence"] = confidence
        if recency_days is not None:
            payload["recency_days"] = recency_days
        
        if not payload:
            logger.warning("No fields to update")
            return False
        
        # updated_ts is automatically set by database default
        response = client.table("context_fact")\
            .update(payload)\
            .eq("fact_id", fact_id)\
            .execute()
        
        if response.data:
            logger.info(f"Successfully updated context fact {fact_id}")
            return True
        else:
            logger.warning(f"No rows updated for fact_id {fact_id}")
            return False
    except Exception as e:
        logger.error(f"Failed to update context fact {fact_id}: {e}")
        return False


def write_user_context_bundle(
    client: Client,
    user_id: str,
    persona_summary: Optional[str] = None,
    last_session_summary: Optional[str] = None,
    facts: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Write or update a user context bundle in the user_context_bundle table.
    This uses upsert (insert or update) since user_id is the primary key.
    
    Args:
        client: Supabase client instance
        user_id: UUID of the user
        persona_summary: Optional persona summary text
        last_session_summary: Optional last session summary text
        facts: Optional list of fact dictionaries (will be stored as JSONB)
    
    Returns:
        True if write/update succeeded, False otherwise
    """
    try:
        payload = {
            "user_id": user_id
        }
        
        if persona_summary is not None:
            payload["persona_summary"] = persona_summary
        if last_session_summary is not None:
            payload["last_session_summary"] = last_session_summary
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


def read_user_context_bundle(
    client: Client,
    user_id: str
) -> Optional[Dict[str, Any]]:
    """
    Read a user's context bundle from the user_context_bundle table.
    
    Args:
        client: Supabase client instance
        user_id: UUID of the user
    
    Returns:
        Dictionary with keys: user_id, version_ts, persona_summary, 
        last_session_summary, facts. Returns None if not found.
    """
    try:
        response = client.table("user_context_bundle")\
            .select("*")\
            .eq("user_id", user_id)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Found context bundle for user {user_id}")
            return response.data[0]
        else:
            logger.info(f"No context bundle found for user {user_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to read context bundle for user {user_id}: {e}")
        return None


def read_user_context_facts(
    client: Client,
    user_id: str,
    limit: int = 100,
    tags: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Read context facts for a specific user.
    
    Args:
        client: Supabase client instance
        user_id: UUID of the user
        limit: Maximum number of facts to return
        tags: Optional list of tags to filter by (finds facts containing any of these tags)
    
    Returns:
        List of fact dictionaries with keys: fact_id, user_id, text, tags, 
        confidence, recency_days, created_ts, updated_ts
    """
    try:
        query = client.table("context_fact")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_ts", desc=True)\
            .limit(limit)
        
        # Note: Supabase array filtering - if tags provided, filter facts that contain any of these tags
        # This is a simplified approach; for exact tag matching, you'd need to use array overlap operator
        response = query.execute()
        
        facts = response.data or []
        
        # Filter by tags if provided (client-side filtering for simplicity)
        if tags:
            filtered_facts = []
            for fact in facts:
                fact_tags = fact.get("tags", [])
                if any(tag in fact_tags for tag in tags):
                    filtered_facts.append(fact)
            facts = filtered_facts
        
        logger.info(f"Found {len(facts)} context facts for user {user_id}")
        return facts
    except Exception as e:
        logger.error(f"Failed to read context facts for user {user_id}: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Example: Connect and perform operations
    try:
        # Get client
        client = get_supabase_client()
        
        # Example user ID (replace with actual user ID)
        example_user_id = "8517c97f-66ef-4955-86ed-531013d33d3e"
        
        print(f"\n=== Reading conversations for user {example_user_id} ===")
        conversations = read_user_conversations(client, example_user_id, limit=5)
        for conv in conversations:
            print(f"Conversation {conv['id']}: Started at {conv.get('started_at')}")
        
        # Read messages from first conversation
        if conversations:
            conv_id = conversations[0]['id']
            print(f"\n=== Reading messages from conversation {conv_id} ===")
            messages = read_conversation_messages(client, conv_id, limit=10)
            for msg in messages[:5]:  # Show first 5
                role = msg.get('role', 'unknown')
                text_preview = msg.get('text', '')[:50]
                print(f"[{role}]: {text_preview}...")
        
        # Write context fact
        print(f"\n=== Writing context fact ===")
        fact_id = write_context_fact(
            client,
            example_user_id,
            text="User prefers morning meditation sessions",
            tags=["preference", "meditation", "schedule"],
            confidence=0.9
        )
        print(f"Created fact with ID: {fact_id}")
        
        # Read context facts
        print(f"\n=== Reading context facts ===")
        facts = read_user_context_facts(client, example_user_id, limit=5)
        for fact in facts:
            print(f"Fact {fact.get('fact_id')}: {fact.get('text')[:50]}...")
        
        # Write context bundle
        print(f"\n=== Writing user context bundle ===")
        success = write_user_context_bundle(
            client,
            example_user_id,
            persona_summary="Active user interested in mindfulness and stress management",
            last_session_summary="Discussed morning routine and meditation preferences",
            facts=[{"fact_id": fact_id, "text": "User prefers morning meditation"}]
        )
        print(f"Context bundle write: {'Success' if success else 'Failed'}")
        
        # Read context bundle
        print(f"\n=== Reading user context bundle ===")
        bundle = read_user_context_bundle(client, example_user_id)
        if bundle:
            print(f"Persona: {bundle.get('persona_summary', 'N/A')[:50]}...")
            print(f"Last session: {bundle.get('last_session_summary', 'N/A')[:50]}...")
        
        print("\n=== Example completed successfully ===")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Exception occurred")

