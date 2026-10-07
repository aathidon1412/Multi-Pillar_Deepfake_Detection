"""
backend/models/requests.py
==========================
Pydantic models for incoming API requests across all 5 analytical pillars.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    """Schema for file upload metadata."""
    original_filename: str
    content_type: Optional[str] = None
    file_size_bytes: Optional[int] = None


class AnalysisRequest(BaseModel):
    """Schema for triggering video analysis."""
    video_id: str
    extract_suspicious_frames: bool = True
    priority: Optional[str] = "normal"


class UniversalAnalysisRequest(BaseModel):
    """Schema for multi-pillar universal analysis (Image, Video, Audio, Document)."""
    p4_benford_enabled: bool = True
    audio_mode: str = Field(default="spoken", description="'spoken' or 'music'")
    generate_xai: bool = True


class AudioAnalysisRequest(BaseModel):
    """Schema for Pillar 3 acoustic deepfake inspection."""
    mode: str = Field(default="spoken", description="'spoken' for speech, 'music' for tracks")


class DocumentAnalysisRequest(BaseModel):
    """Schema for Pillar 4 document forensics."""
    ocr_language: str = "eng"
    is_pdf: bool = False
