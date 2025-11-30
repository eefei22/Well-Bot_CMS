"""
Title Generator Module

This module generates meaningful, concise titles for journal entries using LLM.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from utils.llm import DeepSeekClient

# Load environment variables from .env file
load_dotenv()

# Setup logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language name (e.g., "English", "Chinese", "Malay")
    """
    # Simple heuristic: check for Chinese characters
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if has_chinese:
        return "Chinese"
    
    # Check for common Chinese punctuation
    chinese_punct = ['，', '。', '？', '！', '；', '：']
    if any(punct in text for punct in chinese_punct):
        return "Chinese"
    
    # Default to English (can be enhanced with langdetect library if needed)
    return "English"


def generate_journal_title(body: str) -> str:
    """
    Generate a meaningful, concise title for a journal entry.
    
    Uses DeepSeek LLM to analyze the journal content and generate a title
    that meaningfully describes the content in the same language.
    
    Args:
        body: Journal entry body text
        
    Returns:
        Generated title (max 100 characters, trimmed if needed)
        
    Raises:
        ValueError: If API key is missing or body is empty
        Exception: If LLM API call fails
    """
    # Load API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    if not body or not body.strip():
        raise ValueError("Journal body cannot be empty")
    
    logger.info(f"Generating title for journal entry (length: {len(body)} chars)")
    
    # Detect language from content
    detected_language = detect_language(body)
    logger.debug(f"Detected language: {detected_language}")
    
    # Truncate body if too long (to avoid token limits and costs)
    # Keep first 2000 characters which should be enough for title generation
    body_preview = body[:2000] if len(body) > 2000 else body
    if len(body) > 2000:
        logger.debug(f"Truncated body from {len(body)} to {len(body_preview)} chars for title generation")
    
    # Create prompt for title generation
    language_instruction = ""
    if detected_language == "Chinese":
        language_instruction = "请用中文生成标题。"
    elif detected_language == "English":
        language_instruction = "Generate the title in English."
    else:
        language_instruction = f"Generate the title in {detected_language}."
    
    prompt = f"""You are a helpful assistant that generates concise, meaningful titles for journal entries.

Journal Entry Content:
{body_preview}

Task: Generate a single, concise title that meaningfully describes the main theme or content of this journal entry.

Requirements:
- {language_instruction}
- The title should be concise (preferably under 10 words or 20 characters for Chinese)
- The title should capture the main theme, emotion, or topic discussed
- Do not include quotation marks, dates, or timestamps
- Do not include phrases like "Journal entry about" or "My thoughts on"
- Output only the title text, nothing else

Title:"""

    # Initialize LLM client with faster model for simple tasks
    logger.info("Initializing DeepSeek client for title generation")
    client = DeepSeekClient(api_key=api_key, model="deepseek-chat", timeout=30.0)
    
    # Generate title using LLM
    logger.info("Generating title with LLM")
    try:
        messages_for_llm = [{"role": "user", "content": prompt}]
        generated_title = client.chat(messages_for_llm)
        
        if not generated_title:
            raise ValueError("LLM returned empty title")
        
        # Clean up the title (remove quotes, extra whitespace)
        title = generated_title.strip()
        # Remove surrounding quotes if present
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        title = title.strip()
        
        # Enforce max length (100 chars)
        if len(title) > 100:
            title = title[:100].rstrip()
            logger.warning(f"Title truncated to 100 characters: {title}")
        
        logger.info(f"Generated title: '{title}' (length: {len(title)} chars)")
        
        return title
        
    except Exception as e:
        logger.error(f"Failed to generate title with LLM: {e}")
        raise

