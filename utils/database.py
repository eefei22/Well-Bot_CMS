"""
Database Script

This script handles Supabase database connection and read operations.
Combines database connection and message loading functionality.
"""

import os
from typing import Dict, List, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone, timedelta

# Load environment variables from .env file
load_dotenv()

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_malaysia_timezone():
    """
    Get Malaysia timezone (UTC+8) object.
    Tries zoneinfo first, falls back to pytz, then manual offset.
    
    Returns:
        Timezone object for Asia/Kuala_Lumpur (UTC+8)
    """
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Kuala_Lumpur")
    except (ImportError, Exception):
        # ZoneInfoNotFoundError, ImportError, or other issues - fall back to pytz
        try:
            import pytz
            return pytz.timezone("Asia/Kuala_Lumpur")
        except ImportError:
            # Final fallback: manual UTC+8 offset
            return timezone(timedelta(hours=8))


def get_current_time_utc8() -> datetime:
    """
    Get current time in UTC+8 (Malaysia timezone).
    
    Returns:
        Datetime object with UTC+8 timezone
    """
    malaysia_tz = get_malaysia_timezone()
    return datetime.now(malaysia_tz)


def parse_database_timestamp(timestamp_str: str) -> datetime:
    """
    Parse a database timestamp string, assuming it's stored in UTC+8.
    Database stores timestamps as timezone-naive in UTC+8.
    
    Args:
        timestamp_str: ISO format timestamp string from database
    
    Returns:
        Datetime object with UTC+8 timezone
    """
    malaysia_tz = get_malaysia_timezone()
    
    # Parse timestamp (assuming it's in ISO format, timezone-naive)
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    
    # If timezone-naive, assume it's UTC+8 (database timezone)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=malaysia_tz)
    else:
        # If it has timezone info, convert to UTC+8
        timestamp = timestamp.astimezone(malaysia_tz)
    
    return timestamp


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
    
    This function can be called by context_extractor.py to retrieve user messages.
    
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


def write_users_context_bundle(user_id: str, persona_summary: str = None, facts: str = None) -> bool:
    """
    Write or update a user context bundle in the users_context_bundle table.
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
        response = client.table("users_context_bundle")\
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


def fetch_recent_emotion_logs(user_id: str, hours: int = 48) -> List[Dict]:
    """
    Fetch recent emotion logs for a user from the emotional_log table.
    
    Args:
        user_id: UUID of the user
        hours: Number of hours to look back (default: 48)
    
    Returns:
        List of emotion log dictionaries, each containing:
        - id: integer
        - user_id: uuid
        - timestamp: timestamp without time zone (stored in UTC+8)
        - emotion_label: string ('Angry', 'Sad', 'Happy', 'Fear')
        - confidence_score: float (0.0 to 1.0)
        - emotional_score: integer (0 to 100) or None
    """
    try:
        client = get_supabase_client()
        from datetime import timedelta
        
        # Calculate cutoff time (hours ago from now) in UTC+8
        cutoff_time = get_current_time_utc8() - timedelta(hours=hours)
        
        response = client.table("emotional_log")\
            .select("id, user_id, timestamp, emotion_label, confidence_score, emotional_score")\
            .eq("user_id", user_id)\
            .gte("timestamp", cutoff_time.isoformat())\
            .order("timestamp", desc=False)\
            .execute()
        
        logs = response.data
        logger.info(f"Fetched {len(logs)} emotion logs for user {user_id} (last {hours} hours)")
        return logs
    except Exception as e:
        logger.error(f"Failed to fetch emotion logs for user {user_id}: {e}")
        return []


def get_latest_emotion_log(user_id: str) -> Optional[Dict]:
    """
    Get the latest emotion log entry for a user.
    
    Args:
        user_id: UUID of the user
    
    Returns:
        Dictionary with latest emotion log data or None if not found.
        Contains: id, user_id, timestamp, emotion_label, confidence_score, emotional_score
    """
    try:
        client = get_supabase_client()
        
        response = client.table("emotional_log")\
            .select("id, user_id, timestamp, emotion_label, confidence_score, emotional_score")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            latest = response.data[0]
            logger.info(f"Fetched latest emotion log for user {user_id}: {latest.get('emotion_label')} "
                       f"(confidence: {latest.get('confidence_score'):.2f})")
            return latest
        else:
            logger.info(f"No emotion logs found for user {user_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch latest emotion log for user {user_id}: {e}")
        return None


def fetch_recent_activity_logs(user_id: str, hours: int = 24) -> List[Dict]:
    """
    Fetch recent activity logs for a user from the intervention_log table.
    
    Args:
        user_id: UUID of the user
        hours: Number of hours to look back (default: 24)
    
    Returns:
        List of activity log dictionaries, each containing:
        - id: bigint
        - public_id: uuid
        - user_id: uuid
        - emotional_log_id: bigint or None
        - intervention_type: string ('journal', 'gratitude', 'meditation', 'quote')
        - timestamp: timestamp without time zone (stored in UTC+8)
        - duration: interval or None
    """
    try:
        client = get_supabase_client()
        from datetime import timedelta
        
        # Calculate cutoff time (hours ago from now) in UTC+8
        cutoff_time = get_current_time_utc8() - timedelta(hours=hours)
        
        response = client.table("intervention_log")\
            .select("id, public_id, user_id, emotional_log_id, intervention_type, timestamp, duration")\
            .eq("user_id", user_id)\
            .gte("timestamp", cutoff_time.isoformat())\
            .order("timestamp", desc=False)\
            .execute()
        
        logs = response.data
        logger.info(f"Fetched {len(logs)} activity logs for user {user_id} (last {hours} hours)")
        return logs
    except Exception as e:
        logger.error(f"Failed to fetch activity logs for user {user_id}: {e}")
        return []


def fetch_user_preferences(user_id: str) -> Dict:
    """
    Fetch user preferences from the users table, specifically the prefer_intervention JSONB field.
    
    Args:
        user_id: UUID of the user
    
    Returns:
        Dictionary with preference flags for each activity type.
        Default structure: {"plan": bool, "music": bool, "quote": bool, "converse": bool, 
                           "breathing": bool, "gratitude": bool, "journaling": bool}
        Returns default preferences if user not found or field missing.
    """
    try:
        client = get_supabase_client()
        
        response = client.table("users")\
            .select("prefer_intervention")\
            .eq("id", user_id)\
            .single()\
            .execute()
        
        if response.data and "prefer_intervention" in response.data:
            preferences = response.data["prefer_intervention"]
            logger.info(f"Fetched preferences for user {user_id}: {preferences}")
            return preferences
        else:
            # Return default preferences
            default_prefs = {
                "plan": True,
                "music": True,
                "quote": True,
                "converse": True,
                "breathing": True,
                "gratitude": True,
                "journaling": True
            }
            logger.warning(f"No preferences found for user {user_id}, using defaults")
            return default_prefs
    except Exception as e:
        logger.error(f"Failed to fetch preferences for user {user_id}: {e}")
        # Return default preferences on error
        return {
            "plan": True,
            "music": True,
            "quote": True,
            "converse": True,
            "breathing": True,
            "gratitude": True,
            "journaling": True
        }


def check_embedding_exists(ref_id: str, model_tag: str) -> bool:
    """
    Check if an embedding already exists for a given ref_id and model_tag.
    Used for idempotence - prevents duplicate embeddings.
    
    Args:
        ref_id: Reference ID (message ID or chunk ID)
        model_tag: Model tag ('miniLM' or 'e5')
    
    Returns:
        True if embedding exists, False otherwise
    """
    try:
        client = get_supabase_client()
        
        response = client.table("wb_embeddings")\
            .select("id")\
            .eq("ref_id", ref_id)\
            .eq("model_tag", model_tag)\
            .limit(1)\
            .execute()
        
        exists = len(response.data) > 0
        return exists
        
    except Exception as e:
        logger.error(f"Failed to check embedding existence for ref_id {ref_id}, model_tag {model_tag}: {e}")
        return False


def store_embedding(
    user_id: str,
    kind: str,
    ref_id: str,
    vector: List[float],
    model_tag: str
) -> bool:
    """
    Store an embedding vector in the wb_embeddings table.
    
    Args:
        user_id: UUID of the user
        kind: Type of embedding ('message', 'journal', 'todo', 'preference', 'gratitude')
        ref_id: Reference ID (message ID or chunk ID)
        vector: Embedding vector as list of floats
        model_tag: Model tag ('miniLM' or 'e5')
    
    Returns:
        True if storage succeeded, False otherwise
    """
    try:
        client = get_supabase_client()
        
        # Convert vector to pgvector format (string representation)
        # Supabase pgvector expects format: "[0.1,0.2,...]"
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        
        payload = {
            "user_id": user_id,
            "kind": kind,
            "ref_id": ref_id,
            "vector": vector_str,
            "model_tag": model_tag
        }
        
        response = client.table("wb_embeddings")\
            .insert(payload)\
            .execute()
        
        if response.data:
            logger.debug(f"Successfully stored embedding for ref_id {ref_id}, model_tag {model_tag}")
            return True
        else:
            logger.warning(f"No data returned for embedding storage for ref_id {ref_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to store embedding for ref_id {ref_id}, model_tag {model_tag}: {e}")
        return False


def load_conversation_messages(conversation_id: str) -> List[Dict]:
    """
    Load all user messages for a specific conversation.
    
    Args:
        conversation_id: UUID of the conversation
    
    Returns:
        List of message dictionaries, each containing:
        - id: Message UUID
        - text: Message text
        - created_at: Timestamp
        - role: Message role (should be "user")
    """
    try:
        client = get_supabase_client()
        
        response = client.table("wb_message")\
            .select("id, text, created_at, role")\
            .eq("conversation_id", conversation_id)\
            .eq("role", "user")\
            .order("created_at", desc=False)\
            .execute()
        
        messages = response.data
        logger.info(f"Loaded {len(messages)} user messages for conversation {conversation_id}")
        return messages
        
    except Exception as e:
        logger.error(f"Failed to load messages for conversation {conversation_id}: {e}")
        return []


def get_all_users() -> List[str]:
    """
    Get all user IDs from the database.
    Used for migration purposes.
    
    Returns:
        List of user UUIDs
    """
    try:
        client = get_supabase_client()
        
        response = client.table("users")\
            .select("id")\
            .execute()
        
        user_ids = [user["id"] for user in response.data]
        logger.info(f"Retrieved {len(user_ids)} users from database")
        return user_ids
        
    except Exception as e:
        logger.error(f"Failed to get all users: {e}")
        return []


def get_time_since_last_activity(user_id: str) -> float:
    """
    Calculate the time (in minutes) since the last activity for a user.
    Database timestamps are stored in UTC+8 (timezone-naive).
    
    Args:
        user_id: UUID of the user
    
    Returns:
        Number of minutes since last activity. Returns float('inf') if no activities found.
    """
    try:
        client = get_supabase_client()
        
        # Get the most recent activity
        response = client.table("intervention_log")\
            .select("timestamp")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            last_timestamp_str = response.data[0]["timestamp"]
            # Parse timestamp as UTC+8 (database timezone)
            last_timestamp = parse_database_timestamp(last_timestamp_str)
            
            # Get current time in UTC+8
            now = get_current_time_utc8()
            time_diff = now - last_timestamp
            minutes = time_diff.total_seconds() / 60.0
            logger.info(f"Time since last activity for user {user_id}: {minutes:.2f} minutes")
            return minutes
        else:
            logger.info(f"No activities found for user {user_id}, returning infinity")
            return float('inf')
    except Exception as e:
        logger.error(f"Failed to get time since last activity for user {user_id}: {e}")
        return float('inf')
