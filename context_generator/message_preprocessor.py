"""
Message Preprocessor Module

This module handles database querying and message preprocessing for context generation.
Loads user messages from the database, filters, and normalizes them for LLM processing.
"""

import logging
import re
import uuid
from typing import List, Dict, Tuple

from utils import database
from utils.embeddings import generate_embedding

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
    Normalize a message: lowercase (for non-CJK), strip whitespace, remove extra spaces.
    For CJK languages, skip lowercasing as they don't have case.
    
    Args:
        text: Message text to normalize
    
    Returns:
        Normalized message text
    """
    # Strip whitespace and remove extra spaces
    normalized = ' '.join(text.strip().split())
    
    # Only lowercase if text doesn't contain CJK characters
    if not _has_cjk_characters(normalized):
        normalized = normalized.lower()
    
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
    [DEPRECATED] Load user messages from database and preprocess them for LLM consumption.
    
    This function is deprecated. Use embed_conversation_messages() instead for the new
    embedding-based architecture.
    
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


def _chunk_message(text: str, threshold: int = 500) -> List[Tuple[str, int]]:
    """
    Split long messages into chunks if they exceed the threshold.
    Attempts to split at sentence boundaries when possible.
    
    Args:
        text: Message text to chunk
        threshold: Character threshold for chunking (default: 500)
    
    Returns:
        List of tuples: (chunk_text, chunk_index)
        If no chunking needed, returns [(text, 0)]
    """
    if len(text) <= threshold:
        return [(text, 0)]
    
    chunks = []
    
    # Try to split at sentence boundaries first
    # Match sentence endings: . ! ? followed by space or end of string
    sentence_pattern = r'([.!?]+\s+|$)'
    sentences = re.split(sentence_pattern, text)
    
    # Reconstruct sentences (pattern includes delimiters)
    reconstructed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            reconstructed_sentences.append(sentences[i] + sentences[i + 1])
        else:
            reconstructed_sentences.append(sentences[i])
    
    # If we have sentence boundaries, try to group them into chunks
    if len(reconstructed_sentences) > 1:
        current_chunk = ""
        chunk_index = 0
        
        for sentence in reconstructed_sentences:
            # If adding this sentence would exceed threshold, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) > threshold:
                chunks.append((current_chunk.strip(), chunk_index))
                current_chunk = sentence
                chunk_index += 1
            else:
                current_chunk += sentence
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), chunk_index))
    else:
        # No sentence boundaries found, split at character threshold
        for i in range(0, len(text), threshold):
            chunk = text[i:i + threshold]
            if chunk.strip():
                chunks.append((chunk.strip(), len(chunks)))
    
    return chunks if chunks else [(text, 0)]


def embed_conversation_messages(user_id: str, conversation_id: str, model_tag: str = 'e5') -> Dict:
    """
    Embed user messages from a specific conversation.
    
    This function:
    1. Fetches user messages for the conversation
    2. Filters and normalizes messages
    3. Chunks long messages (>500 chars)
    4. Generates embeddings for each chunk
    5. Stores embeddings in wb_embeddings table (with idempotence check)
    
    Args:
        user_id: UUID of the user
        conversation_id: UUID of the conversation
        model_tag: Model tag for embeddings ('miniLM' or 'e5'), default 'e5'
    
    Returns:
        Dictionary with metadata:
        {
            "messages_processed": int,
            "chunks_created": int,
            "embeddings_stored": int,
            "messages_skipped": int
        }
    
    Raises:
        ValueError: If no messages found or invalid model_tag
    """
    if model_tag not in ['miniLM', 'e5']:
        raise ValueError(f"Invalid model_tag: {model_tag}. Must be 'miniLM' or 'e5'")
    
    # Load messages for this conversation
    logger.info(f"Loading messages for conversation {conversation_id} (user {user_id})")
    messages = database.load_conversation_messages(conversation_id)
    
    if not messages:
        logger.warning(f"No messages found for conversation {conversation_id}")
        return {
            "messages_processed": 0,
            "chunks_created": 0,
            "embeddings_stored": 0,
            "messages_skipped": 0
        }
    
    # Extract message texts
    message_texts = [msg["text"] for msg in messages if msg.get("text")]
    
    if not message_texts:
        raise ValueError(f"No message texts found for conversation {conversation_id}")
    
    logger.info(f"Found {len(message_texts)} messages for conversation {conversation_id}")
    
    # Filter messages (discard short ones)
    filtered_messages = _filter_messages(message_texts, min_words=4)
    
    if not filtered_messages:
        logger.warning(f"No messages with sufficient length for conversation {conversation_id}")
        return {
            "messages_processed": 0,
            "chunks_created": 0,
            "embeddings_stored": 0,
            "messages_skipped": 0
        }
    
    logger.info(f"After filtering: {len(filtered_messages)} messages")
    
    # Normalize messages
    normalized_messages = [_normalize_message(msg) for msg in filtered_messages]
    
    # Process each message: chunk, check idempotence, generate embeddings
    messages_processed = 0
    chunks_created = 0
    embeddings_stored = 0
    messages_skipped = 0
    
    for i, normalized_text in enumerate(normalized_messages):
        message_id = messages[i]["id"]
        
        # Chunk message if needed
        chunks = _chunk_message(normalized_text, threshold=500)
        chunks_created += len(chunks)
        
        # Process each chunk
        for chunk_text, chunk_index in chunks:
            # For chunked messages, create a unique ref_id
            # Generate deterministic UUID from message_id and chunk_index
            if len(chunks) > 1:
                # Create deterministic UUID v5 from message_id (as namespace) + chunk_index
                # This ensures same chunk always gets same ref_id for idempotence
                namespace = uuid.UUID(message_id)
                ref_id = str(uuid.uuid5(namespace, str(chunk_index)))
            else:
                ref_id = message_id
            
            # Check if embedding already exists (idempotence)
            if database.check_embedding_exists(ref_id, model_tag):
                logger.debug(f"Embedding already exists for ref_id {ref_id}, skipping")
                messages_skipped += 1
                continue
            
            # Generate embedding
            try:
                vector = generate_embedding(chunk_text, model_tag=model_tag)
                
                # Store embedding
                success = database.store_embedding(
                    user_id=user_id,
                    kind="message",
                    ref_id=ref_id,
                    vector=vector,
                    model_tag=model_tag
                )
                
                if success:
                    embeddings_stored += 1
                    logger.debug(f"Stored embedding for ref_id {ref_id}")
                else:
                    logger.warning(f"Failed to store embedding for ref_id {ref_id}")
                    
            except Exception as e:
                logger.error(f"Failed to generate/store embedding for ref_id {ref_id}: {e}")
                # Continue with next chunk instead of failing entirely
                continue
        
        messages_processed += 1
    
    logger.info(
        f"Completed embedding for conversation {conversation_id}: "
        f"{messages_processed} messages processed, {chunks_created} chunks created, "
        f"{embeddings_stored} embeddings stored, {messages_skipped} skipped"
    )
    
    return {
        "messages_processed": messages_processed,
        "chunks_created": chunks_created,
        "embeddings_stored": embeddings_stored,
        "messages_skipped": messages_skipped
    }

