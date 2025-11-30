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
from context_generator import context_extractor, facts_extractor, message_preprocessor, title_generator
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
    0. Embeds new conversation messages (if conversation_id provided)
    1. Extracts persona facts (communication style, interests, personality traits, etc.)
    2. Extracts daily life context (stories, routines, relationships, work, etc.)
    3. Saves both to the users_context_bundle table
    
    Args:
        request: ProcessContextRequest containing user_id and optional conversation_id
    
    Returns:
        ProcessContextResponse with facts and persona_summary
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("=== REQUEST RECEIVED ===")
    logger.info(f"User ID: {request.user_id}")
    if request.conversation_id:
        logger.info(f"Conversation ID: {request.conversation_id}")
    else:
        logger.info("Conversation ID: None (skipping incremental embedding)")
    logger.info(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Determine the actual user_id to use
    # If conversation_id is provided, fetch user_id from conversation (source of truth)
    # Otherwise, use the provided user_id
    actual_user_id = request.user_id
    if request.conversation_id:
        try:
            actual_user_id = database.get_conversation_user_id(request.conversation_id)
            logger.info("")
            logger.info(f"[Validation] Conversation {request.conversation_id} belongs to user {actual_user_id}")
            if actual_user_id != request.user_id:
                logger.warning(
                    f"[Validation] Mismatch detected: Request user_id ({request.user_id}) "
                    f"does not match conversation owner ({actual_user_id}). "
                    f"Using conversation owner ({actual_user_id}) as source of truth."
                )
        except ValueError as e:
            logger.error(f"[Validation] Failed to get user_id from conversation: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    # Track durations for final summary
    embed_duration = 0.0
    facts_duration = 0.0
    context_duration = 0.0
    embed_result = None
    facts = None
    persona_summary = None
    
    try:
        # Step 0: Embed new conversation messages (if conversation_id provided)
        if request.conversation_id:
            embed_start = datetime.now()
            logger.info("")
            logger.info("[Step 0] Embedding Messages")
            logger.info(f"  Conversation: {request.conversation_id}")
            logger.info(f"  User: {actual_user_id}")
            try:
                embed_result = message_preprocessor.embed_conversation_messages(
                    conversation_id=request.conversation_id,
                    user_id=actual_user_id  # Pass for validation
                )
                embed_end = datetime.now()
                embed_duration = (embed_end - embed_start).total_seconds()
                logger.info(f"  → Messages processed: {embed_result.get('messages_processed', 0)}")
                logger.info(f"  → Chunks created: {embed_result.get('chunks_created', 0)}")
                logger.info(f"  → Embeddings stored: {embed_result.get('embeddings_stored', 0)}")
                logger.info(f"  → Messages skipped: {embed_result.get('messages_skipped', 0)}")
                logger.info(f"  ✓ Completed in {embed_duration:.2f}s")
            except Exception as e:
                embed_end = datetime.now()
                embed_duration = (embed_end - embed_start).total_seconds()
                logger.error(f"  ✗ Failed after {embed_duration:.2f}s: {e}")
                # Don't raise - continue with extraction even if embedding fails
                # (old embeddings will still be queried)
        else:
            logger.info("")
            logger.info("[Step 0] Embedding Messages")
            logger.info("  → Skipped (no conversation_id provided)")
        
        # Step 1: Extract persona facts (using semantic vector search)
        facts_start = datetime.now()
        logger.info("")
        logger.info("[Step 1] Extracting Persona Facts")
        logger.info(f"  User: {actual_user_id}")
        logger.info("  → Queried 6 focus areas (communication style, interests, personality traits, values, characteristics, behavioural patterns)")
        try:
            facts = facts_extractor.extract_user_facts(actual_user_id)
            facts_end = datetime.now()
            facts_duration = (facts_end - facts_start).total_seconds()
            facts_length = len(facts) if facts else 0
            facts_preview = facts[:200] + "..." if facts and len(facts) > 200 else (facts if facts else "")
            logger.info(f"  → Facts summary: {facts_length:,} characters")
            if facts_preview:
                logger.info(f"  → Preview: {facts_preview}")
            logger.info(f"  ✓ Completed in {facts_duration:.2f}s")
        except Exception as e:
            facts_end = datetime.now()
            facts_duration = (facts_end - facts_start).total_seconds()
            logger.error(f"  ✗ Failed after {facts_duration:.2f}s: {e}")
            # Don't raise - allow context extraction to proceed
        
        # Step 2: Extract daily life context (using semantic vector search)
        context_start = datetime.now()
        logger.info("")
        logger.info("[Step 2] Extracting Daily Life Context")
        logger.info(f"  User: {actual_user_id}")
        logger.info("  → Queried 6 focus areas (routines, stories, relationships, work, events, activities)")
        try:
            persona_summary = context_extractor.process_user_context(actual_user_id)
            context_end = datetime.now()
            context_duration = (context_end - context_start).total_seconds()
            context_length = len(persona_summary) if persona_summary else 0
            context_preview = persona_summary[:200] + "..." if persona_summary and len(persona_summary) > 200 else (persona_summary if persona_summary else "")
            logger.info(f"  → Context summary: {context_length:,} characters")
            if context_preview:
                logger.info(f"  → Preview: {context_preview}")
            logger.info(f"  ✓ Completed in {context_duration:.2f}s")
        except Exception as e:
            context_end = datetime.now()
            context_duration = (context_end - context_start).total_seconds()
            logger.error(f"  ✗ Failed after {context_duration:.2f}s: {e}")
            raise
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Determine overall status
        facts_success = facts is not None
        context_success = persona_summary is not None
        
        if facts_success and context_success:
            overall_status = "Success"
        elif facts_success or context_success:
            overall_status = "Partial Success"
        else:
            overall_status = "Failure"
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Status: {overall_status}")
        logger.info(f"Extracted: Facts {'✓' if facts_success else '✗'} | Context {'✓' if context_success else '✗'}")
        logger.info(f"Total duration: {total_duration:.2f}s")
        if embed_duration > 0:
            logger.info(f"  - Embedding: {embed_duration:.2f}s")
        logger.info(f"  - Facts extraction: {facts_duration:.2f}s")
        logger.info(f"  - Context extraction: {context_duration:.2f}s")
        logger.info("=" * 60)
        
        return schemas.ProcessContextResponse(
            status="success",
            user_id=request.user_id,
            facts=facts,
            persona_summary=persona_summary
        )
    except ValueError as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error("")
        logger.error("=" * 60)
        logger.error("=== PROCESSING FAILED ===")
        logger.error(f"Error Type: ValueError")
        logger.error(f"Error: {e}")
        logger.error(f"Total duration: {total_duration:.2f}s")
        logger.error("=" * 60)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error("")
        logger.error("=" * 60)
        logger.error("=== PROCESSING FAILED ===")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error: {e}")
        logger.error(f"Total duration: {total_duration:.2f}s")
        # Log partial success if any step completed
        if facts_duration > 0 or context_duration > 0:
            logger.error(f"Partial results: Facts {'✓' if facts is not None else '✗'} | Context {'✓' if persona_summary is not None else '✗'}")
        logger.error("=" * 60)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/intervention/suggest", response_model=schemas.SuggestionResponse)
async def suggest_intervention(request: schemas.SuggestionRequest):
    """
    Suggest intervention activities based on emotion and user history.
    
    This endpoint:
    1. Accepts user_id only
    2. Fetches latest emotion from database for the user
    3. Fetches recent emotion logs and activity history for the user
    4. Determines if an intervention should be triggered (kick-start decision)
    5. Generates ranked activity suggestions (1-4) with scores
    6. Returns both the decision and suggestions
    
    Args:
        request: SuggestionRequest containing user_id
    
    Returns:
        SuggestionResponse with decision (trigger_intervention, confidence) and 
        suggestion (ranked activities with scores and reasoning)
    """
    start_time = datetime.now()
    logger.info(f"POST /api/intervention/suggest - Endpoint called at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  User ID: {request.user_id}")
    
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


@app.post("/api/journal/generate-title", response_model=schemas.GenerateTitleResponse)
async def generate_journal_title(request: schemas.GenerateTitleRequest):
    """
    Generate a meaningful, concise title for a journal entry.
    
    This endpoint:
    1. Accepts journal body text
    2. Uses LLM to generate a title that meaningfully describes the content
    3. Returns the generated title in the same language as the journal content
    
    Args:
        request: GenerateTitleRequest containing journal body text
    
    Returns:
        GenerateTitleResponse with generated title
    """
    start_time = datetime.now()
    logger.info(f"POST /api/journal/generate-title - Endpoint called at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Body length: {len(request.body)} chars")
    
    try:
        # Validate request
        if not request.body or not request.body.strip():
            raise HTTPException(status_code=400, detail="Journal body cannot be empty")
        
        # Generate title using title generator
        generated_title = title_generator.generate_journal_title(request.body)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"POST /api/journal/generate-title - Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (total: {total_duration:.2f}s)")
        logger.info(f"  Generated title: '{generated_title}'")
        
        return schemas.GenerateTitleResponse(title=generated_title)
        
    except ValueError as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error(f"POST /api/journal/generate-title - ValueError at {end_time.strftime('%H:%M:%S')} after {total_duration:.2f}s: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.error(f"POST /api/journal/generate-title - Exception at {end_time.strftime('%H:%M:%S')} after {total_duration:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

