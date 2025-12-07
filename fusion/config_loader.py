"""
Configuration Loader for Fusion Service

This module loads configuration from JSON file and provides fallback defaults.
"""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default configuration (used as fallback)
DEFAULT_CONFIG = {
    "fusion_weights": {
        "speech": 0.4,
        "face": 0.3,
        "vitals": 0.3
    },
    "time_window_seconds": 60,
    "model_timeout_seconds": 1.5,
    "model_service_urls": {
        "ser": "http://localhost:8001",
        "fer": "http://localhost:8002",
        "vitals": "http://localhost:8003"
    }
}

# Cache for loaded config
_config_cache: Dict[str, Any] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default path relative to this module.
    
    Returns:
        Configuration dictionary. Returns default config if file not found or invalid.
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache is not None:
        return _config_cache
    
    # Determine config file path
    if config_path is None:
        # Default to config.json in same directory as this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(module_dir, "config.json")
    
    # Try to load config file
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}, using default configuration")
            config = DEFAULT_CONFIG.copy()
        
        # Override with environment variables if set (for cloud deployments)
        # Check for SER_SERVICE_URL, FER_SERVICE_URL, VITALS_SERVICE_URL
        if "model_service_urls" not in config:
            config["model_service_urls"] = {}
        
        ser_url = os.getenv("SER_SERVICE_URL")
        fer_url = os.getenv("FER_SERVICE_URL")
        vitals_url = os.getenv("VITALS_SERVICE_URL")
        
        if ser_url:
            config["model_service_urls"]["ser"] = ser_url
            logger.info(f"Overriding SER URL from environment: {ser_url}")
        if fer_url:
            config["model_service_urls"]["fer"] = fer_url
            logger.info(f"Overriding FER URL from environment: {fer_url}")
        if vitals_url:
            config["model_service_urls"]["vitals"] = vitals_url
            logger.info(f"Overriding Vitals URL from environment: {vitals_url}")
        
        _config_cache = config
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}. Using default configuration.")
        _config_cache = DEFAULT_CONFIG
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}. Using default configuration.")
        _config_cache = DEFAULT_CONFIG
        return DEFAULT_CONFIG



