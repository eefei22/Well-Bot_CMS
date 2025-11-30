"""
Configuration Loader for Intervention Service

This module loads configuration from JSON file and provides fallback defaults.
"""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default configuration (used as fallback)
DEFAULT_CONFIG = {
    "decision_engine": {
        "negative_emotions": ["Sad", "Angry", "Fear"],
        "confidence_threshold": 0.70,
        "min_time_since_last_activity_minutes": 60.0
    },
    "suggestion_engine": {
        "frequency_multipliers": {
            "1": 1.3,
            "2": 1.2,
            "3": 1.1,
            "4": 1.05
        },
        "preference_multipliers": {
            "preferred": 1.2,
            "not_preferred": 0.7
        },
        "activity_types": ["journal", "gratitude", "meditation", "quote"],
        "emotion_activity_weights": {
            "Sad": {
                "journal": 0.9,
                "meditation": 0.8,
                "gratitude": 0.7,
                "quote": 0.6
            },
            "Angry": {
                "meditation": 0.9,
                "journal": 0.7,
                "quote": 0.6,
                "gratitude": 0.5
            },
            "Fear": {
                "meditation": 0.8,
                "quote": 0.7,
                "journal": 0.7,
                "gratitude": 0.6
            },
            "Happy": {
                "gratitude": 0.8,
                "journal": 0.7,
                "quote": 0.6,
                "meditation": 0.5
            }
        }
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
            _config_cache = config
            return config
        else:
            logger.warning(f"Config file not found at {config_path}, using default configuration")
            _config_cache = DEFAULT_CONFIG
            return DEFAULT_CONFIG
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}. Using default configuration.")
        _config_cache = DEFAULT_CONFIG
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}. Using default configuration.")
        _config_cache = DEFAULT_CONFIG
        return DEFAULT_CONFIG

