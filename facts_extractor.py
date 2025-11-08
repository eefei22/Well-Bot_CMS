"""
Facts Extractor Script

This script retrieves and extracts key user persona facts from the database.
Extracts communication style, interests, personality traits, values, and concerns.
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

import database
from llm import DeepSeekClient

# Load environment variables from .env file
load_dotenv()

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _filter_messages(messages: List[str], min_words: int = 4) -> List[str]:
    """
    Filter messages to keep only those with at least min_words words.
    
    Args:
        messages: List of message texts
        min_words: Minimum number of words required (default: 4)
    
    Returns:
        Filtered list of messages
    """
    filtered = []
    for msg in messages:
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


def extract_user_facts(user_id: str) -> str:
    """
    Extract user persona facts and characteristics from messages.
    
    This function:
    1. Loads all user messages from the database
    2. Filters and normalizes messages
    3. Uses DeepSeek reasoning model to extract persona characteristics
    4. Saves the facts to users_context_bundle table (facts field)
    
    Args:
        user_id: UUID of the user
    
    Returns:
        Generated persona facts summary string
    
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
    
    # Create prompt for persona facts extraction
    prompt = f"""You are an intelligent user-profiling assistant.  
            Analyze the following user messages and extract their stable persona characteristics and factual context.

            Focus on these categories:  
            - Communication style and patterns  
            - Interests and preferences  
            - Personality traits  
            - Values and concerns  
            - Notable characteristics  
            - Behavioural patterns  

            User Messages:  
            {messages_text}

            Please output a **structured**, **detailed** summary of the user’s facts in clearly-labelled bullet points by category.  
            Important constraints:  
            • Only include items you can reasonably infer.  
            • Do **not** include additional justification or long explanations.  

            Output only the summary.
            """

    # Initialize LLM client with longer timeout for reasoning model
    logger.info("Initializing DeepSeek client (reasoning model may take 1-3 minutes)")
    client = DeepSeekClient(api_key=api_key, model="deepseek-reasoner", timeout=180.0)
    
    # Generate persona facts using LLM
    logger.info("Generating persona facts with LLM")
    try:
        messages_for_llm = [{"role": "user", "content": prompt}]
        facts_summary = client.chat(messages_for_llm)
        
        if not facts_summary:
            raise ValueError("LLM returned empty facts summary")
        
        logger.info(f"Generated persona facts (length: {len(facts_summary)} chars)")
        
    except Exception as e:
        logger.error(f"Failed to generate persona facts with LLM: {e}")
        raise
    
    # Save facts to database
    logger.info("Saving persona facts to database")
    success = database.write_users_context_bundle(user_id, facts=facts_summary)
    
    if not success:
        logger.warning(f"Failed to save persona facts to database for user {user_id}")
        # Still return the facts even if save failed
    else:
        logger.info(f"Successfully saved persona facts for user {user_id}")
    
    return facts_summary
