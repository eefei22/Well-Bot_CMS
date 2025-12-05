"""
Pydantic Models for Fusion Service

This module defines request/response models and internal data structures
for the emotion fusion service.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class EmotionSnapshotOptions(BaseModel):
    """Optional configuration for emotion snapshot request."""
    timeout_seconds: Optional[float] = Field(default=None, description="Override default timeout for model calls")
    window_seconds: Optional[int] = Field(default=None, description="Override default time window for signal filtering")


class EmotionSnapshotRequest(BaseModel):
    """Request model for emotion snapshot endpoint."""
    user_id: str = Field(..., description="UUID of the user")
    timestamp: Optional[str] = Field(default=None, description="Snapshot timestamp (ISO format). If not provided, uses current time.")
    context_id: Optional[str] = Field(default=None, description="Optional session/conversation ID for linking")
    options: Optional[EmotionSnapshotOptions] = Field(default=None, description="Optional overrides for timeout/window")


class ModelSignal(BaseModel):
    """Model prediction signal structure."""
    user_id: str
    timestamp: str  # ISO format timestamp
    modality: str  # "speech" | "face" | "vitals"
    emotion_label: str  # "Angry" | "Sad" | "Happy" | "Fear"
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class ModelPredictResponse(BaseModel):
    """Response structure from model service /predict endpoint."""
    signals: List[ModelSignal] = Field(default_factory=list, description="List of predictions within the time window")


class SignalUsed(BaseModel):
    """Signal used in fusion (for response)."""
    modality: str
    emotion_label: str
    confidence: float


class FusedEmotionResponse(BaseModel):
    """Response model for emotion snapshot endpoint."""
    user_id: str
    timestamp: str  # ISO format timestamp
    emotion_label: str  # "Angry" | "Sad" | "Happy" | "Fear"
    confidence_score: float = Field(ge=0.0, le=1.0, description="Fused confidence score")
    emotional_score: int = Field(ge=0, le=100, description="Emotional score mapped to 0-100 scale")
    signals_used: List[SignalUsed] = Field(default_factory=list, description="Signals that contributed to fusion")


class NoSignalsResponse(BaseModel):
    """Response when no valid signals are available."""
    status: str = "no_signals"
    reason: str = "no valid modality outputs"



