"""
Pydantic Models for Intervention Service

This module defines request/response models and internal data structures
for the intervention suggestion service.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class SuggestionRequest(BaseModel):
    """Request model for intervention suggestion endpoint."""
    user_id: str


class RankedActivity(BaseModel):
    """Model for a ranked activity suggestion."""
    activity_type: str  # 'journal', 'gratitude', 'meditation', 'quote'
    rank: int  # 1-4 (1 is best)
    score: float  # Strength of suggestion (0.0 to 1.0)


class DecisionResult(BaseModel):
    """Result from decision engine (kick-start decision)."""
    trigger_intervention: bool
    confidence_score: float  # Confidence in the decision (0.0 to 1.0)
    reasoning: Optional[str] = None  # Optional reasoning for the decision


class SuggestionResult(BaseModel):
    """Result from suggestion engine (activity recommendations)."""
    ranked_activities: List[RankedActivity]  # All activities ranked 1-4
    reasoning: Optional[str] = None  # Optional reasoning for the suggestions


class SuggestionResponse(BaseModel):
    """Response model for intervention suggestion endpoint."""
    user_id: str
    decision: DecisionResult
    suggestion: SuggestionResult

