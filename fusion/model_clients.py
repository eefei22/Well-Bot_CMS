"""
Model Client Layer for Fusion Service

This module provides HTTP clients for communicating with SER, FER, and Vitals model services.
Each client implements retry logic and graceful error handling.
"""

import httpx
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fusion.config_loader import load_config
from fusion.models import ModelSignal, ModelPredictResponse

logger = logging.getLogger(__name__)

# Load configuration
_config = load_config()
_model_config = _config.get("model_service_urls", {})
_timeout_config = _config.get("model_timeout_seconds", 1.5)


class BaseModelClient:
    """Base class for model service clients with retry logic."""
    
    def __init__(self, service_url: str, service_name: str, timeout: float = None):
        """
        Initialize base model client.
        
        Args:
            service_url: Base URL of the model service
            service_name: Name of the service (for logging)
            timeout: Request timeout in seconds (defaults to config value)
        """
        self.service_url = service_url.rstrip("/")
        self.service_name = service_name
        self.timeout = timeout or _timeout_config
        self.predict_endpoint = f"{self.service_url}/predict"
        
        logger.info(f"{service_name}Client initialized with URL: {self.service_url}")
    
    async def predict(
        self,
        user_id: str,
        snapshot_timestamp: datetime,
        window_seconds: int = 60
    ) -> List[ModelSignal]:
        """
        Get predictions from the model service within the specified time window.
        
        Args:
            user_id: UUID of the user
            snapshot_timestamp: Timestamp for the snapshot (UTC+8 timezone-aware)
            window_seconds: Time window in seconds (default: 60)
        
        Returns:
            List of ModelSignal objects, or empty list on failure
        """
        # Convert timestamp to ISO format string
        timestamp_str = snapshot_timestamp.isoformat()
        
        # Prepare request payload
        payload = {
            "user_id": user_id,
            "snapshot_timestamp": timestamp_str,
            "window_seconds": window_seconds
        }
        
        # Try with retry logic (1 retry)
        result = await self._make_request(payload)
        if result is not None:
            return result
        
        # Retry once if first attempt failed
        logger.info(f"Retrying {self.service_name} prediction request...")
        result = await self._make_request(payload)
        if result is not None:
            return result
        
        logger.warning(f"{self.service_name} prediction failed after retry, returning empty list")
        return []
    
    async def _make_request(self, payload: Dict[str, Any]) -> Optional[List[ModelSignal]]:
        """
        Make HTTP request to model service.
        
        Args:
            payload: Request payload dictionary
        
        Returns:
            List of ModelSignal objects if successful, None on failure
        """
        try:
            timeout_config = httpx.Timeout(
                connect=5.0,
                read=self.timeout,
                write=5.0,
                pool=5.0
            )
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    self.predict_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                predict_response = ModelPredictResponse(**data)
                
                logger.debug(f"{self.service_name} returned {len(predict_response.signals)} signals")
                return predict_response.signals
                
        except httpx.TimeoutException:
            logger.warning(f"{self.service_name} request timed out after {self.timeout}s")
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f"{self.service_name} returned HTTP {e.response.status_code}: {e}")
            return None
        except Exception as e:
            logger.warning(f"{self.service_name} request failed: {e}")
            return None


class SERClient(BaseModelClient):
    """Client for Speech Emotion Recognition (SER) service."""
    
    def __init__(self, service_url: Optional[str] = None, timeout: Optional[float] = None):
        """
        Initialize SER client.
        
        Args:
            service_url: Optional service URL override
            timeout: Optional timeout override
        """
        url = service_url or _model_config.get("ser", "http://localhost:8001")
        super().__init__(url, "SER", timeout)


class FERClient(BaseModelClient):
    """Client for Face Emotion Recognition (FER) service."""
    
    def __init__(self, service_url: Optional[str] = None, timeout: Optional[float] = None):
        """
        Initialize FER client.
        
        Args:
            service_url: Optional service URL override
            timeout: Optional timeout override
        """
        url = service_url or _model_config.get("fer", "http://localhost:8002")
        super().__init__(url, "FER", timeout)


class VitalsClient(BaseModelClient):
    """Client for Vitals Emotion Recognition service."""
    
    def __init__(self, service_url: Optional[str] = None, timeout: Optional[float] = None):
        """
        Initialize Vitals client.
        
        Args:
            service_url: Optional service URL override
            timeout: Optional timeout override
        """
        url = service_url or _model_config.get("vitals", "http://localhost:8003")
        super().__init__(url, "Vitals", timeout)



