"""
Pydantic Schemas / Data Transfer Objects (DTOs)

This script defines request/response models for the API.
"""

from pydantic import BaseModel
from typing import Optional

# Re-export intervention models for API use
from intervention.models import (
    SuggestionRequest,
    SuggestionResponse
)


class ProcessContextRequest(BaseModel):
    """Request model for processing user context."""
    user_id: str


class ProcessContextResponse(BaseModel):
    """Response model for processing user context."""
    status: str
    user_id: str
    facts: Optional[str] = None
    persona_summary: Optional[str] = None

