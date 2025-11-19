"""
Vector Search Module

This module provides semantic similarity search functionality using Supabase pgvector.
Supports querying embeddings by semantic similarity and retrieving original message texts.
"""

import logging
from typing import List, Dict, Set
from utils import database
from utils.embeddings import generate_query_embedding

# Setup logging
logger = logging.getLogger(__name__)


def search_similar_embeddings(
    user_id: str,
    query_vector: List[float],
    model_tag: str,
    similarity_threshold: float = 0.7,
    kind: str = 'message',
    match_limit: int = None,
    index_limit: int = None
) -> List[Dict]:
    """
    Perform cosine similarity search using pgvector via Supabase RPC.
    
    Queries the wb_embeddings table filtered by user_id, model_tag, and kind.
    Returns results above similarity threshold (dynamic, no fixed limit by default).
    
    Args:
        user_id: UUID of the user
        query_vector: Query embedding vector (list of floats)
        model_tag: Model tag ('miniLM' or 'e5')
        similarity_threshold: Minimum similarity score (0.0-1.0), default 0.7
        kind: Type of embedding ('message', 'journal', etc.), default 'message'
        match_limit: Maximum number of results to return (None = no limit)
        index_limit: Number of nearest neighbors to fetch from HNSW index (None = use default)
    
    Returns:
        List of dicts, each containing:
        - ref_id: UUID reference ID
        - similarity_score: Similarity score (0.0-1.0)
        - kind: Embedding kind
        - created_at: Timestamp
    
    Raises:
        Exception: If RPC call fails or database error occurs
    """
    try:
        client = database.get_supabase_client()
        
        # Build RPC parameters
        # Always include match_limit and index_limit to use the optimized version with HNSW index
        # This helps Supabase choose the correct overloaded function when both versions exist
        rpc_params = {
            'query_vector': query_vector,  # Pass as list, Supabase handles conversion
            'match_user_id': user_id,
            'match_model_tag': model_tag,
            'match_kind': kind,
            'match_threshold': similarity_threshold,
            'match_limit': match_limit,  # None = no limit (dynamic), gets all results above threshold
            'index_limit': index_limit if index_limit is not None else 100  # Default 100 for HNSW index efficiency
        }
        
        # Call RPC function for vector similarity search
        response = client.rpc('match_embeddings', rpc_params).execute()
        
        if response.data:
            # Transform response to match expected format
            results = []
            for item in response.data:
                results.append({
                    'ref_id': item.get('ref_id'),
                    'similarity_score': item.get('similarity'),
                    'kind': item.get('kind'),
                    'created_at': item.get('created_at')
                })
            
            logger.debug(
                f"Found {len(results)} similar embeddings for user {user_id} "
                f"(threshold: {similarity_threshold}, kind: {kind})"
            )
            return results
        else:
            logger.debug(
                f"No similar embeddings found for user {user_id} "
                f"(threshold: {similarity_threshold}, kind: {kind})"
            )
            return []
            
    except Exception as e:
        logger.error(f"Failed to search similar embeddings: {e}")
        raise


def query_embeddings_by_semantic_prompt(
    user_id: str,
    query_text: str,
    model_tag: str = 'e5',
    similarity_threshold: float = 0.7,
    kind: str = 'message'
) -> List[Dict]:
    """
    Generate embedding for query text and perform semantic similarity search.
    
    Args:
        user_id: UUID of the user
        query_text: Semantic query text (e.g., "daily routines and activities")
        model_tag: Model tag ('miniLM' or 'e5'), default 'e5'
        similarity_threshold: Minimum similarity score (0.0-1.0), default 0.7
        kind: Type of embedding ('message', 'journal', etc.), default 'message'
    
    Returns:
        List of dicts with ref_id, similarity_score, kind, created_at
        (same format as search_similar_embeddings)
    
    Raises:
        ValueError: If query_text is empty
        Exception: If embedding generation or search fails
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")
    
    logger.debug(f"Generating query embedding for: '{query_text}'")
    
    # Generate query embedding (uses "query: " prefix for E5 model)
    query_vector = generate_query_embedding(query_text, model_tag=model_tag)
    
    # Perform similarity search
    return search_similar_embeddings(
        user_id=user_id,
        query_vector=query_vector,
        model_tag=model_tag,
        similarity_threshold=similarity_threshold,
        kind=kind
    )


def retrieve_message_texts(ref_ids: List[str]) -> Dict[str, str]:
    """
    Fetch original message texts from wb_message table using ref_ids.
    
    Args:
        ref_ids: List of message UUIDs (ref_ids from embeddings)
    
    Returns:
        Dict mapping ref_id -> message text
        Only includes ref_ids that were found in the database
    
    Raises:
        Exception: If database query fails
    """
    if not ref_ids:
        return {}
    
    try:
        client = database.get_supabase_client()
        
        # Query wb_message table for all ref_ids
        # Use 'in' filter to get multiple messages at once
        response = client.table("wb_message")\
            .select("id, text")\
            .in_("id", ref_ids)\
            .execute()
        
        # Build mapping dictionary
        message_texts = {}
        if response.data:
            for msg in response.data:
                msg_id = msg.get("id")
                msg_text = msg.get("text", "")
                if msg_id and msg_text:
                    message_texts[str(msg_id)] = msg_text
        
        logger.debug(f"Retrieved {len(message_texts)} message texts from {len(ref_ids)} ref_ids")
        return message_texts
        
    except Exception as e:
        logger.error(f"Failed to retrieve message texts: {e}")
        raise

