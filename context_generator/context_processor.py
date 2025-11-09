"""
Context Processor Script

This script retrieves and processes user's daily life context from the database.
Extracts stories, experiences, routines, relationships, work life, and people interactions.
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

from utils import database
from utils.llm import DeepSeekClient

# Load environment variables from .env file
load_dotenv()

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _filter_messages(messages: List[str], min_words: int = 4) -> List[str]:
    """
    Filter messages to keep only those with at least min_words words.
    For Chinese/CJK text, counts characters instead of words.
    
    Args:
        messages: List of message texts
        min_words: Minimum number of words required (default: 4)
    
    Returns:
        Filtered list of messages
    """
    def _has_cjk_characters(text: str) -> bool:
        """Check if text contains Chinese, Japanese, or Korean characters."""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Chinese
                return True
            if '\u3040' <= char <= '\u309f':  # Hiragana
                return True
            if '\u30a0' <= char <= '\u30ff':  # Katakana
                return True
            if '\uac00' <= char <= '\ud7a3':  # Korean
                return True
        return False
    
    filtered = []
    for msg in messages:
        # For Chinese/CJK text, count characters instead of words
        if _has_cjk_characters(msg):
            # Count non-whitespace characters for CJK languages
            char_count = len([c for c in msg if not c.isspace()])
            if char_count >= min_words:
                filtered.append(msg)
            else:
                logger.debug(f"Filtered out short CJK message: {msg[:50]}... (chars: {char_count})")
        else:
            # For languages with spaces, count words
            word_count = len(msg.split())
            if word_count >= min_words:
                filtered.append(msg)
            else:
                logger.debug(f"Filtered out short message: {msg[:50]}... (words: {word_count})")
    return filtered


def _normalize_message(text: str) -> str:
    """
    Normalize a message: lowercase, strip whitespace, remove extra spaces.
    
    Args:
        text: Message text to normalize
    
    Returns:
        Normalized message text
    """
    # Lowercase, strip, and remove extra spaces
    normalized = ' '.join(text.lower().strip().split())
    return normalized


def _extract_message_texts(conversations: List[Dict]) -> List[str]:
    """
    Extract all message texts from the conversation structure.
    
    Args:
        conversations: List of conversation dictionaries from load_user_messages
    
    Returns:
        List of message text strings
    """
    messages = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            text = msg.get("text", "")
            if text:
                messages.append(text)
    return messages


def process_user_context(user_id: str) -> str:
    """
    Process user messages to generate and save a daily life context summary.
    
    This function:
    1. Loads all user messages from the database
    2. Filters and normalizes messages
    3. Uses DeepSeek reasoning model to extract daily life stories and experiences
    4. Saves the context summary to users_context_bundle table (persona_summary field)
    
    Args:
        user_id: UUID of the user
    
    Returns:
        Generated daily life context summary string
    
    Raises:
        ValueError: If API key is missing or no messages found
        Exception: If LLM API call fails
    """
    # Load API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    # Load user messages
    logger.info(f"Loading messages for user {user_id}")
    conversations = database.load_user_messages(user_id)
    
    if not conversations:
        raise ValueError(f"No conversations found for user {user_id}")
    
    # Extract all message texts
    all_messages = _extract_message_texts(conversations)
    
    if not all_messages:
        raise ValueError(f"No messages found for user {user_id}")
    
    logger.info(f"Extracted {len(all_messages)} total messages")
    
    # Filter messages (discard very short ones)
    filtered_messages = _filter_messages(all_messages, min_words=4)
    
    if not filtered_messages:
        raise ValueError(f"No messages with sufficient length found for user {user_id}")
    
    logger.info(f"After filtering: {len(filtered_messages)} messages")
    
    # Normalize messages
    normalized_messages = [_normalize_message(msg) for msg in filtered_messages]
    
    # Format messages for LLM prompt
    messages_text = "\n".join([f"- {msg}" for msg in normalized_messages])
    
    # Create prompt for daily life context extraction
    prompt = f"""You are an intelligent context-extraction assistant.  
            Analyze the following user messages and extract the key daily-life context, background and experiential stories of the user.

            Focus on extracting:  
            - Daily routines and activities  
            - Stories and experiences the user shares  
            - People they meet and their relationships  
            - Work life and professional context  
            - Life events and significant moments  
            - Day-to-day activities and interactions  

            User Messages:  
            {messages_text}

            Please output a **structured**, **detailed** summary of the user’s daily-life context in **clearly-labelled bullet points** grouped by category.  
            Constraints:  
            • Produce the summary in the same language as the user messages.
            • Only include items supported by the messages; avoid speculation.  
            • Use relative timestamps if available (e.g., “recently”, “over past month”).  
            • Do **not** include extra commentary or reflection or labelling — just the summary.

            Output only the summary.

            """

    # Initialize LLM client with longer timeout for reasoning model
    logger.info("Initializing DeepSeek client (reasoning model may take 1-3 minutes)")
    client = DeepSeekClient(api_key=api_key, model="deepseek-reasoner", timeout=180.0)
    
    # Generate context summary using LLM
    logger.info("Generating daily life context summary with LLM")
    try:
        messages_for_llm = [{"role": "user", "content": prompt}]
        context_summary = client.chat(messages_for_llm)
        
        if not context_summary:
            raise ValueError("LLM returned empty context summary")
        
        logger.info(f"Generated context summary (length: {len(context_summary)} chars)")
        
    except Exception as e:
        logger.error(f"Failed to generate context summary with LLM: {e}")
        raise
    
    # Save context summary to database (stored in persona_summary field)
    logger.info("Saving context summary to database")
    success = database.write_users_context_bundle(user_id, persona_summary=context_summary)
    
    if not success:
        logger.warning(f"Failed to save context summary to database for user {user_id}")
        # Still return the context summary even if save failed
    else:
        logger.info(f"Successfully saved context summary for user {user_id}")
    
    return context_summary

