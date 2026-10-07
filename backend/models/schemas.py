"""
backend/models/schemas.py
=========================
Pydantic schemas re-exported from backend.models.requests and backend.models.responses
for seamless backward compatibility.
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
]
