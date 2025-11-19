"""
Facts Extractor Script

This script extracts key user persona facts from preprocessed messages.
Extracts communication style, interests, personality traits, values, and concerns.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv

from utils import database
from utils.llm import DeepSeekClient

# Load environment variables from .env file
load_dotenv()

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_user_facts(user_id: str, preprocessed_messages: List[str]) -> str:
    """
    Extract user persona facts and characteristics from preprocessed messages.
    
    This function:
    1. Uses DeepSeek reasoning model to extract persona characteristics from preprocessed messages
    2. Saves the facts to users_context_bundle table (facts field)
    
    Args:
        user_id: UUID of the user
        preprocessed_messages: List of normalized message strings (already filtered and normalized)
    
    Returns:
        Generated persona facts summary string
    
    Raises:
        ValueError: If API key is missing or preprocessed_messages is empty
        Exception: If LLM API call fails
    """
    # Load API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    if not preprocessed_messages:
        raise ValueError("preprocessed_messages cannot be empty")
    
    logger.info(f"Extracting persona facts from {len(preprocessed_messages)} preprocessed messages for user {user_id}")
    
    # Format messages for LLM prompt
    messages_text = "\n".join([f"- {msg}" for msg in preprocessed_messages])
    
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
            • Produce the facts in the same language as the user messages.
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
