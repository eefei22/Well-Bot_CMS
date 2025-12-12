"""
Model Client Layer for Fusion Service

This module provides HTTP clients for communicating with SER, FER, and Vitals model services.
Each client implements retry logic and graceful error handling.
Supports simulation endpoints when demo mode is enabled.
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
_simulation_service_url = _config.get("simulation_service_url", None)


class BaseModelClient:
    """Base class for model service clients with retry logic."""
    
    def __init__(self, service_url: str, service_name: str, timeout: float = None, modality: str = None):
        """
        Initialize base model client.
        
        Args:
            service_url: Base URL of the model service
            service_name: Name of the service (for logging)
            timeout: Request timeout in seconds (defaults to config value)
            modality: Modality name for simulation endpoints ("ser", "fer", "vitals")
        """
        self.service_url = service_url.rstrip("/")
        self.service_name = service_name
        self.timeout = timeout or _timeout_config
        self.modality = modality  # For simulation endpoint mapping
        self.predict_endpoint = f"{self.service_url}/predict"
        
        logger.info(f"{service_name}Client initialized with URL: {self.service_url}")
    
    async def check_demo_mode(self, simulation_service_url: Optional[str] = None) -> bool:
        """
        Check if demo mode is enabled on the simulation service.
        
        Args:
            simulation_service_url: Optional URL override for simulation service
            
        Returns:
            True if demo mode is enabled, False otherwise
        """
        url = simulation_service_url or _simulation_service_url
        if not url:
            logger.debug("No simulation service URL configured, demo mode check skipped")
            return False
        
        try:
            demo_mode_url = f"{url.rstrip('/')}/simulation/demo-mode"
            timeout_config = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.get(demo_mode_url)
                response.raise_for_status()
                data = response.json()
                enabled = data.get("enabled", False)
                logger.debug(f"Demo mode check: {enabled}")
                return enabled
        except Exception as e:
            logger.debug(f"Demo mode check failed (assuming OFF): {e}")
            return False
    
    def _get_simulation_endpoint(self, simulation_service_url: Optional[str] = None) -> str:
        """
        Get simulation endpoint URL for this modality.
        
        Args:
            simulation_service_url: Optional URL override for simulation service
            
        Returns:
            Simulation endpoint URL
        """
        url = simulation_service_url or _simulation_service_url
        if not url:
            raise ValueError("No simulation service URL configured")
        
        if not self.modality:
            raise ValueError(f"No modality set for {self.service_name} client")
        
        # Map modality to simulation endpoint path
        modality_lower = self.modality.lower()
        return f"{url.rstrip('/')}/simulation/{modality_lower}/predict"
    
    async def predict(
        self,
        user_id: str,
        snapshot_timestamp: datetime,
        window_seconds: int = 60
    ) -> List[ModelSignal]:
        """
        Get predictions from the model service within the specified time window.
        Checks demo mode and uses simulation endpoints if enabled.
        
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
        
        # Check demo mode if simulation service URL is configured
        use_simulation = False
        if _simulation_service_url:
            use_simulation = await self.check_demo_mode()
            if use_simulation:
                logger.info(f"{self.service_name} using simulation endpoint (demo mode ON)")
            else:
                logger.debug(f"{self.service_name} using regular endpoint (demo mode OFF)")
        
        # Try with retry logic (1 retry)
        result = await self._make_request(payload, use_simulation=use_simulation)
        if result is not None:
            return result
        
        # Retry once if first attempt failed
        logger.info(f"Retrying {self.service_name} prediction request...")
        result = await self._make_request(payload, use_simulation=use_simulation)
        if result is not None:
            return result
        
        logger.warning(f"{self.service_name} prediction failed after retry, returning empty list")
        return []
    
    async def _make_request(
        self, 
        payload: Dict[str, Any], 
        use_simulation: bool = False
    ) -> Optional[List[ModelSignal]]:
        """
        Make HTTP request to model service.
        
        Args:
            payload: Request payload dictionary
            use_simulation: If True, use simulation endpoint instead of regular endpoint
        
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
            
            # Determine endpoint URL
            if use_simulation:
                endpoint = self._get_simulation_endpoint()
                logger.debug(f"{self.service_name} calling simulation endpoint: {endpoint}")
            else:
                endpoint = self.predict_endpoint
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                predict_response = ModelPredictResponse(**data)
                
                endpoint_type = "simulation" if use_simulation else "regular"
                logger.debug(
                    f"{self.service_name} ({endpoint_type}) returned {len(predict_response.signals)} signals"
                )
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
        super().__init__(url, "SER", timeout, modality="ser")


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
        super().__init__(url, "FER", timeout, modality="fer")


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
        super().__init__(url, "Vitals", timeout, modality="vitals")



