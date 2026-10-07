"""
backend/models/responses.py
===========================
Pydantic schemas for outgoing API responses, status tracking, forensic scoring,
and explainable AI (XAI) across all 5 analytical pillars.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    video_id: str
    status: str
    progress: int
    current_stage: str
    error: Optional[str] = None


class ClassificationScores(BaseModel):
    real: float = 0.0
    ai_generated: float = 0.0
    forged: float = 0.0


class ClassificationResult(BaseModel):
    prediction: str
    confidence: float
    scores: ClassificationScores


class VideoDetails(BaseModel):
    duration_seconds: float
    resolution: str
    fps: float
    frame_count: int


class FileMetadata(BaseModel):
    original_filename: str
    stored_filename: str
    format: str
    size_mb: float


class ConsensusResult(BaseModel):
    verdict: str
    confidence: float
    is_real: bool
    override_reason: Optional[str] = None
    engines: Optional[str] = None


class XAIEvidence(BaseModel):
    feature: str
    value: Any
    direction: str
    importance: float
    description: str


class XAIVisualization(BaseModel):
    title: str
    type: str
    data_url: Optional[str] = None
    description: Optional[str] = None


class XAIBundle(BaseModel):
    xai_available: bool = True
    pillar: str
    prediction: str
    confidence: float
    evidence: List[XAIEvidence] = []
    visualizations: List[XAIVisualization] = []
    human_explanation: str
    technical_explanation: str
    limitations: Optional[List[str]] = None


class PillarResult(BaseModel):
    pillar_name: str
    prediction: str
    confidence: float
    is_real: bool
    details: Dict[str, Any] = {}
    xai: Optional[XAIBundle] = None


class AnalysisReport(BaseModel):
    video_id: str
    modality: str = "video"
    filename: Optional[str] = None
    file: Optional[FileMetadata] = None
    video_details: Optional[VideoDetails] = None
    classification: Optional[ClassificationResult] = None
    consensus: Optional[ConsensusResult] = None
    suspicious_frames: Optional[List[Dict[str, Any]]] = None
    pillar1: Optional[Dict[str, Any]] = None
    pillar2: Optional[Dict[str, Any]] = None
    pillar3: Optional[Dict[str, Any]] = None
    pillar4: Optional[Dict[str, Any]] = None
    pillar5: Optional[Dict[str, Any]] = None
    xai: Optional[Dict[str, Any]] = None
    processing: Optional[Dict[str, Any]] = None
