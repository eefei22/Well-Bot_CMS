"""
Facts Extractor Script

This script extracts key user persona facts using semantic vector search.
Extracts communication style, interests, personality traits, values, and concerns.
"""

import os
import logging
from typing import List, Set
from dotenv import load_dotenv

from utils import database
from utils.llm import DeepSeekClient
from utils import vector_search

# Load environment variables from .env file
load_dotenv()

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_user_facts(user_id: str, model_tag: str = 'e5') -> str:
    """
    Extract user persona facts and characteristics using semantic vector search.
    
    This function:
    1. Performs semantic queries for each focus area (communication style, interests, personality traits, etc.)
    2. Retrieves relevant message texts using vector similarity search
    3. Uses DeepSeek reasoning model to extract persona characteristics from retrieved messages
    4. Saves the facts to users_context_bundle table (facts field)
    
    Args:
        user_id: UUID of the user
        model_tag: Embedding model tag ('miniLM' or 'e5'), default 'e5'
    
    Returns:
        Generated persona facts summary string
    
    Raises:
        ValueError: If API key is missing
        Exception: If vector search or LLM API call fails
    """
    # Load API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    logger.info(f"Extracting persona facts for user {user_id} using semantic search (model: {model_tag})")
    
    # Define focus areas for semantic queries
    focus_areas = [
        "communication style and patterns",
        "interests and preferences",
        "personality traits and characteristics",
        "values and concerns",
        "notable characteristics and traits",
        "behavioural patterns and habits"
    ]
    
    # Perform semantic queries for each focus area
    all_ref_ids: Set[str] = set()
    similarity_threshold = 0.7
    
    logger.info(f"Performing semantic queries for {len(focus_areas)} focus areas")
    for i, focus_area in enumerate(focus_areas, 1):
        try:
            logger.debug(f"Querying focus area {i}/{len(focus_areas)}: '{focus_area}'")
            results = vector_search.query_embeddings_by_semantic_prompt(
                user_id=user_id,
                query_text=focus_area,
                model_tag=model_tag,
                similarity_threshold=similarity_threshold,
                kind='message'
            )
            
            # Collect unique ref_ids
            for result in results:
                ref_id = result.get('ref_id')
                if ref_id:
                    all_ref_ids.add(str(ref_id))
            
            logger.debug(f"Found {len(results)} results for '{focus_area}' (unique ref_ids so far: {len(all_ref_ids)})")
            
        except Exception as e:
            logger.warning(f"Failed to query focus area '{focus_area}': {e}")
            # Continue with other focus areas even if one fails
            continue
    
    # If no results found, try lowering threshold
    if not all_ref_ids:
        logger.warning(f"No results found with threshold {similarity_threshold}, trying lower threshold 0.6")
        similarity_threshold = 0.6
        for focus_area in focus_areas:
            try:
                results = vector_search.query_embeddings_by_semantic_prompt(
                    user_id=user_id,
                    query_text=focus_area,
                    model_tag=model_tag,
                    similarity_threshold=similarity_threshold,
                    kind='message'
                )
                for result in results:
                    ref_id = result.get('ref_id')
                    if ref_id:
                        all_ref_ids.add(str(ref_id))
            except Exception as e:
                logger.warning(f"Failed to query focus area '{focus_area}' with lower threshold: {e}")
                continue
    
    if not all_ref_ids:
        raise ValueError(f"No relevant messages found for user {user_id} with semantic search")
    
    logger.info(f"Retrieved {len(all_ref_ids)} unique message references")
    
    # Retrieve message texts from database
    logger.info("Retrieving message texts from database")
    message_texts = vector_search.retrieve_message_texts(list(all_ref_ids))
    
    if not message_texts:
        raise ValueError(f"Failed to retrieve message texts for user {user_id}")
    
    logger.info(f"Retrieved {len(message_texts)} message texts")
    
    # Format messages for LLM prompt
    messages_text = "\n".join([f"- {text}" for text in message_texts.values()])
    
    # Fetch user's preferred language from database (instead of detecting from messages)
    language_code = database.get_user_language(user_id)
    
    # Map language code/name to human-readable name for LLM prompt
    # Support both codes (en, zh, ms) and full names (English, Chinese, Malay)
    language_map = {
        "en": "English",
        "zh": "Chinese",
        "ms": "Malay",
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "ms-MY": "Malay",
        # Also support full language names directly
        "English": "English",
        "Chinese": "Chinese",
        "Malay": "Malay"
    }
    preferred_language = language_map.get(language_code, "English")  # Default to English if unknown code
    
    # Create prompt for persona facts extraction with STRONG language enforcement
    prompt = f"""You are an intelligent user-profiling assistant.  
            Analyze the following user messages and extract their stable persona characteristics and factual context.

            CRITICAL LANGUAGE REQUIREMENT:
            - The user's preferred language is {preferred_language}
            - You MUST write your entire response ONLY in {preferred_language}
            - IGNORE the language of the user messages below
            - If preferred language is English, respond in English
            - If preferred language is Chinese, respond in Chinese
            - If preferred language is Malay (Bahasa Melayu), respond in Malay
            - Do NOT switch languages under any circumstances

            Focus on these categories:  
            - Communication style and patterns  
            - Interests and preferences  
            - Personality traits  
            - Values and concerns  
            - Notable characteristics  
            - Behavioural patterns  

            User Messages:  
            {messages_text}

            Please output a **structured**, **detailed** summary of the user's facts in clearly-labelled bullet points by category.  
            Important constraints:  
            • Output MUST be in {preferred_language} (the user's preferred language)
            • Only include items you can reasonably infer.  
            • Do **not** include additional justification or long explanations.  

            Output only the summary in {preferred_language}.
            """

    # Initialize LLM client
    logger.info("Initializing DeepSeek client (chat model)")
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
