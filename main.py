"""
FastAPI Main Application

This script orchestrates all routes and runs the FastAPI server on port 8000.
"""

from fastapi import FastAPI, HTTPException
import uvicorn
import logging

# Import modules
import database
import context_processor
import facts_extractor
import schemas

# Setup logging
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
    return {"message": "Well-Bot CMS API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/context/process", response_model=schemas.ProcessContextResponse)
async def process_user_context(request: schemas.ProcessContextRequest):
    """
    Process user messages and generate both persona facts and daily life context.
    
    This endpoint:
    1. Extracts persona facts (communication style, interests, personality traits, etc.)
    2. Extracts daily life context (stories, routines, relationships, work, etc.)
    3. Saves both to the user_context_bundle table
    
    Args:
        request: ProcessContextRequest containing user_id
    
    Returns:
        ProcessContextResponse with facts and persona_summary
    """
    try:
        # Extract persona facts first
        facts = None
        try:
            facts = facts_extractor.extract_user_facts(request.user_id)
        except Exception as e:
            # Log error but continue with context extraction
            logger.error(f"Failed to extract facts for user {request.user_id}: {e}")
            # Don't raise - allow context extraction to proceed
        
        # Extract daily life context
        persona_summary = None
        try:
            persona_summary = context_processor.process_user_context(request.user_id)
        except Exception as e:
            # If context extraction also fails, raise the error
            logger.error(f"Failed to extract context for user {request.user_id}: {e}")
            raise
        
        return schemas.ProcessContextResponse(
            status="success",
            user_id=request.user_id,
            facts=facts,
            persona_summary=persona_summary
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

