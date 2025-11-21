"""
Embedding Generation Client

This module provides embedding generation using sentence-transformers library.
Supports both 'miniLM' and 'e5' models for generating text embeddings.
"""

import logging
from typing import List, Optional
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


def _load_model(model_tag: str):
    """
    Load embedding model with lazy loading and caching.
    
    Args:
        model_tag: Model identifier ('miniLM' or 'e5')
    
    Returns:
        Loaded sentence-transformers model
    """
    if model_tag in _model_cache:
        return _model_cache[model_tag]
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Map model tags to actual model names
        model_map = {
            'miniLM': 'sentence-transformers/all-MiniLM-L6-v2',
            'e5': 'intfloat/multilingual-e5-base'
        }
        
        if model_tag not in model_map:
            raise ValueError(f"Unknown model_tag: {model_tag}. Must be 'miniLM' or 'e5'")
        
        model_name = model_map[model_tag]
        logger.info(f"Loading embedding model: {model_name} (tag: {model_tag})")
        
        model = SentenceTransformer(model_name)
        _model_cache[model_tag] = model
        
        logger.info(f"Successfully loaded model: {model_name}")
        return model
        
    except ImportError:
        raise ImportError(
            "sentence-transformers library not installed. "
            "Install it with: pip install sentence-transformers"
        )


def generate_embedding(text: str, model_tag: str = 'e5') -> List[float]:
    """
    Generate embedding vector for a text string.
    
    Args:
        text: Text string to embed
        model_tag: Model identifier ('miniLM' or 'e5'), default 'e5'
    
    Returns:
        List of floats representing the embedding vector (normalized for cosine similarity)
    
    Raises:
        ValueError: If model_tag is invalid
        ImportError: If sentence-transformers is not installed
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Load model (cached after first load)
    model = _load_model(model_tag)
    
    # For E5 models, prepend "query: " or "passage: " prefix
    # For now, we'll use "passage: " for message embeddings
    if model_tag == 'e5':
        # E5 models require a prefix
        prefixed_text = f"passage: {text}"
    else:
        prefixed_text = text
    
    # Generate embedding
    try:
        embedding = model.encode(prefixed_text, normalize_embeddings=True)
        
        # Convert numpy array to list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def generate_query_embedding(text: str, model_tag: str = 'e5') -> List[float]:
    """
    Generate embedding vector for a query text string.
    
    For E5 models, uses "query: " prefix (different from "passage: " used for storage).
    This is important for optimal semantic search performance.
    
    Args:
        text: Query text string to embed
        model_tag: Model identifier ('miniLM' or 'e5'), default 'e5'
    
    Returns:
        List of floats representing the embedding vector (normalized for cosine similarity)
    
    Raises:
        ValueError: If model_tag is invalid or text is empty
        ImportError: If sentence-transformers is not installed
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Load model (cached after first load)
    model = _load_model(model_tag)
    
    # For E5 models, prepend "query: " prefix (different from "passage: " for storage)
    if model_tag == 'e5':
        prefixed_text = f"query: {text}"
    else:
        prefixed_text = text
    
    # Generate embedding
    try:
        embedding = model.encode(prefixed_text, normalize_embeddings=True)
        
        # Convert numpy array to list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise


def generate_embeddings_batch(texts: List[str], model_tag: str = 'e5', batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts (more efficient than individual calls).
    
    Args:
        texts: List of text strings to embed
        model_tag: Model identifier ('miniLM' or 'e5'), default 'e5'
        batch_size: Batch size for processing (default: 32)
    
    Returns:
        List of embedding vectors (each is a list of floats)
    
    Raises:
        ValueError: If model_tag is invalid or texts is empty
        ImportError: If sentence-transformers is not installed
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    # Load model
    model = _load_model(model_tag)
    
    # For E5 models, prepend "passage: " prefix
    if model_tag == 'e5':
        prefixed_texts = [f"passage: {text}" for text in texts]
    else:
        prefixed_texts = texts
    
    # Generate embeddings in batches
    try:
        embeddings = model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        # Convert numpy arrays to lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise

