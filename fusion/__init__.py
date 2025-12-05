"""
Fusion package for emotion fusion service.

This package provides:
- API endpoints: POST /emotion/snapshot, GET /emotion/health
- Orchestrator: Coordinates the complete fusion flow
- Model clients: HTTP clients for SER/FER/Vitals services
- Fusion logic: Weighted aggregation algorithm
"""

from . import api
from . import orchestrator
from . import model_clients
from . import fusion_logic
from . import models

__all__ = [
    'api',
    'orchestrator',
    'model_clients',
    'fusion_logic',
    'models'
]



