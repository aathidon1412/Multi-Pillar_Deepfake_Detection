"""
backend/models/__init__.py
==========================
Pydantic schemas and ML model interfaces across all analytical pillars.
"""

from backend.models.requests import (
    UploadRequest,
    AnalysisRequest,
    UniversalAnalysisRequest,
    AudioAnalysisRequest,
    DocumentAnalysisRequest,
)

from backend.models.responses import (
    StatusResponse,
    ClassificationScores,
    ClassificationResult,
    VideoDetails,
    FileMetadata,
    ConsensusResult,
    XAIEvidence,
    XAIVisualization,
    XAIBundle,
    PillarResult,
    AnalysisReport,
)

from backend.models.model_interface import (
    BaseVisualModel,
    BaseTemporalModel,
    BaseAudioModel,
    HybridVisualModel,
    DefaultTemporalModel,
    DefaultAudioModel,
    visual_model,
    temporal_model,
    audio_model,
)

__all__ = [
    "UploadRequest",
    "AnalysisRequest",
    "UniversalAnalysisRequest",
    "AudioAnalysisRequest",
    "DocumentAnalysisRequest",
    "StatusResponse",
    "ClassificationScores",
    "ClassificationResult",
    "VideoDetails",
    "FileMetadata",
    "ConsensusResult",
    "XAIEvidence",
    "XAIVisualization",
    "XAIBundle",
    "PillarResult",
    "AnalysisReport",
    "BaseVisualModel",
    "BaseTemporalModel",
    "BaseAudioModel",
    "HybridVisualModel",
    "DefaultTemporalModel",
    "DefaultAudioModel",
    "visual_model",
    "temporal_model",
    "audio_model",
]
