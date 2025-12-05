"""
API Layer for Fusion Service

This module provides FastAPI endpoints for the emotion fusion service.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from fusion.models import EmotionSnapshotRequest, FusedEmotionResponse, NoSignalsResponse
from fusion.orchestrator import process_emotion_snapshot
from utils import database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion", tags=["emotion"])


@router.post("/snapshot", response_model=FusedEmotionResponse)
async def emotion_snapshot(request: EmotionSnapshotRequest):
    """
    Create an emotion snapshot by fusing predictions from SER, FER, and Vitals services.
    
    This endpoint:
    1. Calls model services (SER, FER, Vitals) in parallel
    2. Collects their outputs within the specified time window
    3. Runs fusion logic to compute final emotion
    4. Writes a row into emotional_log table
    5. Returns the fused result
    
    Args:
        request: EmotionSnapshotRequest with user_id and optional parameters
    
    Returns:
        FusedEmotionResponse with fused emotion result, or NoSignalsResponse if no valid signals
    """
    logger.info(f"POST /emotion/snapshot - Endpoint called for user {request.user_id}")
    
    try:
        result = await process_emotion_snapshot(request)
        
        # Check if we got a NoSignalsResponse
        if isinstance(result, NoSignalsResponse):
            logger.warning(f"No signals available for user {request.user_id}: {result.reason}")
            return JSONResponse(
                status_code=200,
                content=result.dict()
            )
        
        # Return successful fused result
        logger.info(f"POST /emotion/snapshot - Successfully processed for user {request.user_id}")
        return result
        
    except ValueError as e:
        logger.error(f"POST /emotion/snapshot - Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"POST /emotion/snapshot - Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health():
    """
    Health check endpoint for fusion service.
    
    Checks:
    - Database connectivity
    
    Returns:
        Dictionary with health status
    """
    logger.info("GET /emotion/health - Health check called")
    
    try:
        # Check database connectivity
        client = database.get_supabase_client()
        # Simple query to test connection
        client.table("emotional_log").select("id").limit(1).execute()
        
        return {
            "status": "healthy",
            "service": "fusion",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "fusion",
                "database": "disconnected",
                "error": str(e)
            }
        )

