"""
Message Preprocessor Module

This module handles database querying and message preprocessing for context generation.
Loads user messages from the database, filters, and normalizes them for LLM processing.
"""

import logging
from typing import List, Dict

from utils import database

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def preprocess_user_messages(user_id: str) -> List[str]:
    """
    Load user messages from database and preprocess them for LLM consumption.
    
    This function:
    1. Loads all user messages from the database (single query)
    2. Extracts message texts from conversation structure
    3. Filters messages (discards short ones)
    4. Normalizes messages (lowercase, strip whitespace)
    5. Returns list of normalized message strings ready for LLM processing
    
    Args:
        user_id: UUID of the user
    
    Returns:
        List of normalized message strings ready for LLM processing
    
    Raises:
        ValueError: If no conversations or messages found, or no messages with sufficient length
    """
    # Load user messages from database
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
    
    logger.info(f"Preprocessed {len(normalized_messages)} messages for LLM processing")
    
    return normalized_messages

