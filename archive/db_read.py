"""
Database Read Script

This script loads user messages from public.wb_message for a specific user.
Can be used as a component or run standalone.

Loads user UUID from .env file (DEV_USER_ID) when run standalone.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from db_connect import get_supabase_client
from supabase import Client

# Load environment variables
load_dotenv()


def load_user_messages(user_id: str) -> list:
    """
    Load all user messages from public.wb_message for a specific user, grouped by conversation.
    
    Args:
        user_id: UUID of the user
    
    Returns:
        List of conversation dictionaries, each containing:
        - user_id: UUID of the user
        - conversation_id: UUID of the conversation
        - conversation_created_at: Timestamp when conversation started
        - total_messages: Count of user messages in this conversation
        - messages: Array of user messages (filtered to role="user" only)
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
        print(f"No conversations found for user {user_id}")
        return []
    
    print(f"Found {len(conversations)} conversations for user {user_id}")
    
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
            print(f"  Conversation {conv_id}: {len(formatted_messages)} user messages")
    
    total_user_messages = sum(conv['total_messages'] for conv in result)
    print(f"\nLoaded {total_user_messages} user messages from {len(result)} conversations")
    return result


if __name__ == "__main__":
    # Get user ID from environment variable
    user_id = os.getenv("DEV_USER_ID")
    
    if not user_id:
        print("Error: DEV_USER_ID environment variable is not set")
        print("Please set DEV_USER_ID in your .env file")
        sys.exit(1)
    
    print(f"Loading messages for user: {user_id}\n")
    
    # Load conversations with user messages
    conversations = load_user_messages(user_id)
    
    # Create output directory if it doesn't exist
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"messages_{user_id}_{timestamp}.json"
    output_path = output_dir / output_filename
    
    # Display results
    if conversations:
        print(f"\n=== Conversations Summary ===")
        total_user_messages = sum(conv['total_messages'] for conv in conversations)
        print(f"Total conversations: {len(conversations)}")
        print(f"Total user messages: {total_user_messages}")
        
        # Show first conversation as example
        if conversations:
            first_conv = conversations[0]
            print(f"\n=== First Conversation Example ===")
            print(f"Conversation ID: {first_conv['conversation_id']}")
            print(f"Created at: {first_conv['conversation_created_at']}")
            print(f"User messages: {first_conv['total_messages']}")
            if first_conv['messages']:
                print(f"First message preview: {first_conv['messages'][0]['text'][:80]}...")
        
        # Save to JSON file - output is array of conversation objects
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Output Saved ===")
        print(f"Conversations saved to: {output_path}")
    else:
        print("No conversations with user messages found for this user")
        
        # Still create an empty output file (empty array)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        print(f"Empty result saved to: {output_path}")

