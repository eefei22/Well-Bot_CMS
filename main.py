"""
FastAPI Main Application

This script orchestrates all routes and runs the FastAPI server on port 8000.
"""

from fastapi import FastAPI, HTTPException
import uvicorn
import logging
from datetime import datetime

# Import modules
from utils import database, schemas
from context_generator import context_processor, facts_extractor
from intervention import intervention

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Well-Bot CMS API",
    description="Context Management System for Well-Bot",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("GET / - Root endpoint called")
    return {"message": "Well-Bot CMS API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    logger.info("GET /health - Health check endpoint called")
    return {"status": "healthy"}


@app.post("/api/context/process", response_model=schemas.ProcessContextResponse)
async def process_user_context(request: schemas.ProcessContextRequest):
    """
    Process user messages and generate both persona facts and daily life context.
    
    This endpoint:
    1. Extracts persona facts (communication style, interests, personality traits, etc.)
    2. Extracts daily life context (stories, routines, relationships, work, etc.)
    3. Saves both to the users_context_bundle table
    
    Args:
        request: ProcessContextRequest containing user_id
    
    Returns:
        ProcessContextResponse with facts and persona_summary
    """
    start_time = datetime.now()
    logger.info(f"POST /api/context/process - Endpoint called at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  User ID: {request.user_id}")
    
    try:
        # Extract persona facts first
        facts = None
        facts_start = datetime.now()
        logger.info(f"  Starting facts extraction at {facts_start.strftime('%H:%M:%S')}")
        try:
            facts = facts_extractor.extract_user_facts(request.user_id)
            facts_end = datetime.now()
            facts_duration = (facts_end - facts_start).total_seconds()
            logger.info(f"  Facts extraction completed at {facts_end.strftime('%H:%M:%S')} (took {facts_duration:.2f}s)")
        except Exception as e:
            facts_end = datetime.now()
            facts_duration = (facts_end - facts_start).total_seconds()
            logger.error(f"  Facts extraction failed at {facts_end.strftime('%H:%M:%S')} after {facts_duration:.2f}s: {e}")
            # Don't raise - allow context extraction to proceed
        
        # Extract daily life context
        persona_summary = None
        context_start = datetime.now()
        logger.info(f"  Starting context extraction at {context_start.strftime('%H:%M:%S')}")
        try:
            persona_summary = context_processor.process_user_context(request.user_id)
            context_end = datetime.now()
            context_duration = (context_end - context_start).total_seconds()
            logger.info(f"  Context extraction completed at {context_end.strftime('%H:%M:%S')} (took {context_duration:.2f}s)")
        except Exception as e:
            context_end = datetime.now()
            context_duration = (context_end - context_start).total_seconds()
            logger.error(f"  Context extraction failed at {context_end.strftime('%H:%M:%S')} after {context_duration:.2f}s: {e}")
            raise
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"POST /api/context/process - Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (total: {total_duration:.2f}s)")
        
        return schemas.ProcessContextResponse(
            status="success",
            user_id=request.user_id,
            facts=facts,
            persona_summary=persona_summary
        )
    except ValueError as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error(f"POST /api/context/process - ValueError at {end_time.strftime('%H:%M:%S')} after {total_duration:.2f}s: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error(f"POST /api/context/process - Exception at {end_time.strftime('%H:%M:%S')} after {total_duration:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/intervention/suggest", response_model=schemas.SuggestionResponse)
async def suggest_intervention(request: schemas.SuggestionRequest):
    """
    Suggest intervention activities based on emotion and user history.
    
    This endpoint:
    1. Fetches recent emotion logs and activity history for the user
    2. Determines if an intervention should be triggered (kick-start decision)
    3. Generates ranked activity suggestions (1-5) with scores
    4. Returns both the decision and suggestions
    
    Args:
        request: SuggestionRequest containing user_id, emotion_label, confidence_score, timestamp
    
    Returns:
        SuggestionResponse with decision (trigger_intervention, confidence) and 
        suggestion (ranked activities with scores and reasoning)
    """
    start_time = datetime.now()
    logger.info(f"POST /api/intervention/suggest - Endpoint called at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  User ID: {request.user_id}, Emotion: {request.emotion_label}, Confidence: {request.confidence_score}")
    
    try:
        response = intervention.process_suggestion_request(request)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"POST /api/intervention/suggest - Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (total: {total_duration:.2f}s)")
        logger.info(f"  Decision: trigger={response.decision.trigger_intervention}, confidence={response.decision.confidence_score:.2f}")
        
        return response
    except ValueError as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error(f"POST /api/intervention/suggest - ValueError at {end_time.strftime('%H:%M:%S')} after {total_duration:.2f}s: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error(f"POST /api/intervention/suggest - Exception at {end_time.strftime('%H:%M:%S')} after {total_duration:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/intervention/health")
async def intervention_health():
    """
    Health check endpoint for intervention service.
    
    Returns:
        Health status with service availability and database connectivity
    """
    try:
        # Test database connectivity
        from utils import database
        client = database.get_supabase_client()
        
        # Simple query to test connection
        test_response = client.table("users").select("id").limit(1).execute()
        
        return {
            "status": "healthy",
            "service": "intervention",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "intervention",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

