"""
Database Connection Script

This script handles Supabase database connection initialization.
"""

import os
from typing import Dict
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supabase_config() -> Dict[str, str]:
    """
    Get Supabase configuration from environment variables.
    
    Returns:
        Dictionary with 'url' and 'service_role_key'
    
    Raises:
        ValueError: If required environment variables are missing
    """
    url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not service_role_key:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")
    
    return {
        "url": url,
        "service_role_key": service_role_key
    }


def get_supabase_client(service: bool = True) -> Client:
    """
    Create and return a Supabase client instance.
    
    Args:
        service: If True, use service_role_key (for admin operations).
                If False, use anon_key (for user-scoped operations).
    
    Returns:
        Supabase Client instance
    """
    config = get_supabase_config()
    url = config["url"]
    key = config["service_role_key"] if service else os.getenv("SUPABASE_ANON_KEY", "")
    
    if not key:
        raise ValueError("Service role key or anon key is required")
    
    client = create_client(url, key)
    logger.info("Successfully connected to Supabase")
    return client

