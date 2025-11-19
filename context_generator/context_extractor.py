"""
Context Extractor Script

This script extracts user's daily life context from preprocessed messages.
Extracts stories, experiences, routines, relationships, work life, and people interactions.
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


def process_user_context(user_id: str, preprocessed_messages: List[str]) -> str:
    """
    Process preprocessed messages to generate and save a daily life context summary.
    
    This function:
    1. Uses DeepSeek reasoning model to extract daily life stories and experiences from preprocessed messages
    2. Saves the context summary to users_context_bundle table (persona_summary field)
    
    Args:
        user_id: UUID of the user
        preprocessed_messages: List of normalized message strings (already filtered and normalized)
    
    Returns:
        Generated daily life context summary string
    
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
    
    logger.info(f"Processing daily life context from {len(preprocessed_messages)} preprocessed messages for user {user_id}")
    
    # Format messages for LLM prompt
    messages_text = "\n".join([f"- {msg}" for msg in preprocessed_messages])
    
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

            Please output a **structured**, **detailed** summary of the user's daily-life context in **clearly-labelled bullet points** grouped by category.  
            Constraints:  
            • Produce the summary in the same language as the user messages.
            • Only include items supported by the messages; avoid speculation.  
            • Use relative timestamps if available (e.g., "recently", "over past month").  
            • Do **not** include extra commentary or reflection or labelling.
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

