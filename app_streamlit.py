"""
================================================================================
Universal Synthetic Media Forensics Engine (USMFE) - Streamlit Web Dashboard
================================================================================
All-in-One Multi-Pillar Deepfake Detection System:
  • Pillar 1: Vision Transformer (ViT) Spectral Neural Artifacts
  • Pillar 2: Video Authenticity (Visual Seams, Temporal Flickering, Lip-Sync, Acoustics)
  • Pillar 3: Voice & Audio Synthetic Speech Forensics (Wav2Vec2 + HPSS Demixing)
  • Pillar 4: Statistical Document, Invoice & Benford's Law OCR Forensics
  • Pillar 5: Hybrid Perspective Geometry & Multi-Generator ML Ensemble (RANSAC)
  • Consensus: Domain-Aware Calibrated Decision Fusion
================================================================================
"""

import io
import os
import sys
import json
import time
import uuid
import tempfile
import traceback
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2
import fitz  # PyMuPDF for PDF documents
import pytesseract
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure sys.path for Pillar 2 backend services
PILLAR2_BACKEND_DIR = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector")
if PILLAR2_BACKEND_DIR not in sys.path:
    sys.path.insert(0, PILLAR2_BACKEND_DIR)

# Core imports
from core import (
    load_pillar1_vit,
    run_pillar1_inference,
    classify_audio,
    run_pillar4_inference,
    load_pillar5_ml_bundle,
    run_pillar5_inference,
    fuse_multi_pillar_verdict,
    apply_custom_theme,
    render_header,
    render_verdict_banner,
    plot_benford_distribution,
)

# Pillar 2 Video Forensics & JSON Engine Imports
try:
    from backend.services.video_processor import inspect_video
    from backend.services.frame_extractor import extract_sampled_frames
    from backend.services.face_detector import detect_faces_in_frames
    from backend.services.visual_analyzer import run_visual_analysis
    from backend.services.temporal_analyzer import run_temporal_analysis
    from backend.services.audio_analyzer import extract_audio_track, run_audio_analysis
    from backend.services.lip_sync_analyzer import run_lip_sync_analysis
    from backend.services.metadata_analyzer import run_metadata_analysis
    from backend.services.classifier import run_feature_fusion_and_classification
    from backend.services.explainability import extract_and_annotate_suspicious_frames
    from backend.services.cleanup import cleanup_temporary_frames
    from backend.services.json_storage import save_result, load_result, list_results
    PILLAR2_AVAILABLE = True
except Exception as e:
    print(f"[Pillar 2 Import Warning]: {e}")
    PILLAR2_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
AUDIO_EXTS = {"wav", "mp3", "flac", "ogg", "m4a"}
PDF_EXTS   = {"pdf"}
ALL_EXTS   = list(IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS | PDF_EXTS)

MODALITY_META = {
    "image": {
        "icon": "🖼️",
        "label": "Visual Photo / Render",
        "color": "#4facfe",
        "pillars": "Pillar 1 (ViT Neural Forensics) + Pillar 5 (Shadow RANSAC Physics)",
    },
    "video": {
        "icon": "🎬",
        "label": "Digital Video Stream",
        "color": "#ec4899",
        "pillars": "Pillar 2 (ViT Artifacts, Temporal Flickering, Acoustics, Lip-Sync)",
    },
    "audio": {
        "icon": "🎧",
        "label": "Audio / Voice Track",
        "color": "#a78bfa",
        "pillars": "Pillar 3 (Acoustic Transformer + HPSS Demixing)",
    },
    "pdf": {
        "icon": "📄",
        "label": "Multi-Page PDF Document",
        "color": "#fb923c",
        "pillars": "Pillar 4 (Benford's Law OCR Statistical Forensics)",
    },
}

# ─── Page Config & Custom Theme ──────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Pillar Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()

# ─── Additional Dynamic CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Modality badge strip */
    .modality-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 50px;
        padding: 8px 18px;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        backdrop-filter: blur(8px);
        margin-bottom: 16px;
    }
    .modality-icon { font-size: 1.1rem; }
    /* Active pillar chips */
    .pillar-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,242,254,0.08);
        border: 1px solid rgba(0,242,254,0.25);
        border-radius: 30px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #00f2fe;
        letter-spacing: 0.3px;
        margin: 3px 4px;
    }
    .pillar-chip.inactive {
        background: rgba(138,153,173,0.06);
        border-color: rgba(138,153,173,0.15);
        color: #8a99ad;
    }
    /* Toggle card */
    .toggle-card {
        background: rgba(18,26,43,0.7);
        border: 1px solid rgba(255,170,0,0.25);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    .toggle-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #ffaa00;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .unified-ai-gen {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(244, 63, 94, 0.03));
        border: 2px solid #f43f5e;
    }
    .unified-forged {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.03));
        border: 2px solid #f59e0b;
    }
    .pillar-card {
        background: rgba(18, 26, 43, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        backdrop-filter: blur(12px);
    }
    .pillar-card--dim {
        opacity: 0.55;
    }
    .pillar-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ffaa00;
        margin-bottom: 4px;
    }
    .metric-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        background: rgba(255, 255, 255, 0.04);
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 6px;
    }
    .upload-hint {
        text-align: center;
        color: #8a99ad;
        font-size: 0.92rem;
        margin-top: 6px;
        font-style: italic;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8a99ad;
        margin-bottom: 10px;
    }
    .audio-card {
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .benford-card {
        background: rgba(251,146,60,0.07);
        border: 1px solid rgba(251,146,60,0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .info-row {
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

render_header()

# ─── Load Cached Core Forensic Models ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_pillar1():
    return load_pillar1_vit()

@st.cache_resource(show_spinner=False)
def get_cached_pillar5():
    return load_pillar5_ml_bundle()

processor_or_tfm, p1_model, device, p1_model_name = get_cached_pillar1()
p5_bundle, p5_model_filename = get_cached_pillar5()


# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Forensic Workspace")

    analysis_mode = st.radio(
        "🎯 Select Forensic Pipeline:",
        [
            "🌐 Universal Multi-Pillar Media Analysis",
            "🎬 Pillar 2: Video Authenticity & Deepfake Forensics",
            "🎙️ Pillar 3: Voice & Audio Synthetic Speech Forensics",
            "📄 Pillar 4: Document, Invoice & PDF Statistical Forensics"
        ],
        index=0
    )
    st.divider()

    # Pillar 4 Benford Toggle (relevant for images with numbers)
    st.markdown("""<div class="toggle-card">
        <div class="toggle-label">🔬 Pillar 4 — Benford's Law OCR</div>
    </div>""", unsafe_allow_html=True)
    p4_enabled = st.toggle(
        "Enable Document / Invoice Analysis",
        value=True,
        help="When ON: visual images that contain numeric text or tabular invoices are tested via Benford's Law."
    )
    st.caption(
        "✅ Active — applies to images with numeric content" if p4_enabled
        else "⛔ Disabled — image analysis skips OCR Benford check"
    )
    st.divider()

    # Audio Sub-Mode selector
    st.markdown("**🎙️ Audio Analysis Sub-Mode**")
    audio_submode = st.radio(
        "Voice / Music pipeline:",
        ["🗣️ Spoken Voice (Standard)", "🎵 Music Track (Experimental / Demixed)"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()

    # Engine Status Indicators
    st.markdown("### 🏛️ Forensic Engine Telemetry")
    st.markdown(f"🔹 **Pillar 1:** ViT Neural Spectral (`{p1_model_name}`)")
    p2_badge = "🟢 Active" if PILLAR2_AVAILABLE else "🔴 Backend Warning"
    st.markdown(f"🔹 **Pillar 2:** Video Authenticity Engine ({p2_badge})")
    st.markdown("🔹 **Pillar 3:** Wav2Vec2 + HPSS Audio Forensics (🟢 Active)")
    p4_badge = "🟢 Active" if p4_enabled else "🔴 Disabled via toggle"
    st.markdown(f"🔹 **Pillar 4:** Benford's Law Statistical OCR ({p4_badge})")
    st.markdown(f"🔹 **Pillar 5:** {p5_model_filename} (🟢 Authoritative)")
    st.divider()

    # Pre-loaded sample repositories
    st.markdown("### 🧪 Pre-loaded Forensic Test Samples")

    sample_category = st.selectbox("Sample Modality", ["Images", "Videos", "Audio", "Documents"])

    IMAGE_SAMPLES = {
        "Select an image sample…": None,
        "AI_img1.jpeg (Fake AI Generated)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img1.jpeg"),
        "AI_img2.jpeg (Fake AI Generated)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img2.jpeg"),
        "AI_img3.png (Fake AI Generated)":  os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img3.png"),
        "AI_img4.png (Fake AI Generated)":  os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img4.png"),
        "AI_img5.png (Fake AI Generated)":  os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img5.png"),
        "Real_img1.JPG (Authentic Photo)":   os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img1.JPG"),
        "Real_img2.jpg (Authentic Photo)":   os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img2.jpg"),
        "Real_img3.jpg (Authentic Photo)":   os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img3.jpg"),
    }

    test_vid_p2 = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector", "storage", "uploads", "VID_TEST_001.mp4")
    real_vid_p2 = os.path.join(BASE_DIR, "Pillar 2", "real_face.mp4")
    fake_vid_p2 = os.path.join(BASE_DIR, "Pillar 2", "fake_avatar.mp4")

    VIDEO_SAMPLES = {
        "Select a video sample…": None,
        "VID_TEST_001.mp4 (Synthetic Video Anomaly)": test_vid_p2 if os.path.exists(test_vid_p2) else None,
        "fake_avatar.mp4 (AI Generated Face)": fake_vid_p2 if os.path.exists(fake_vid_p2) else None,
        "real_face.mp4 (Authentic Human Video)": real_vid_p2 if os.path.exists(real_vid_p2) else None,
    }

    test_audio_path = os.path.join(BASE_DIR, "test_audio", "sample_test.wav")
    ai_voice_path = os.path.join(BASE_DIR, "testing", "AI_Voice.mp3")
    real_voice_path = os.path.join(BASE_DIR, "testing", "Real_voice.mp3")
    AUDIO_SAMPLES = {
        "Select an audio sample…": None,
        "sample_test.wav (Synthetic / Cloned Audio)": test_audio_path if os.path.exists(test_audio_path) else None,
        "AI_Voice.mp3 (AI Synthesized Voice)": ai_voice_path if os.path.exists(ai_voice_path) else None,
        "Real_voice.mp3 (Authentic Human Voice)": real_voice_path if os.path.exists(real_voice_path) else None,
    }

    DOC_SAMPLES = {
        "Select a document sample…": None,
        "invoice_1.png (Authentic Invoice)":  os.path.join(BASE_DIR, "Pillar 4", "invoice_1.png"),
        "invoice_01.png (Authentic Invoice)": os.path.join(BASE_DIR, "Pillar 4", "invoice_01.png"),
        "invoice_2.jpg (Authentic Invoice)":  os.path.join(BASE_DIR, "Pillar 4", "invoice_2.jpg"),
        "invoice_3.png (Authentic Invoice)":  os.path.join(BASE_DIR, "Pillar 4", "invoice_3.png"),
        "invoice_4.png (Authentic Invoice)":  os.path.join(BASE_DIR, "Pillar 4", "invoice_4.png"),
        "invoice_5.png (Authentic Invoice)":  os.path.join(BASE_DIR, "Pillar 4", "invoice_5.png"),
    }

    if sample_category == "Images":
        active_sample_map = IMAGE_SAMPLES
    elif sample_category == "Videos":
        active_sample_map = VIDEO_SAMPLES
    elif sample_category == "Audio":
        active_sample_map = AUDIO_SAMPLES
    else:
        active_sample_map = DOC_SAMPLES

    selected_sample_key = st.selectbox("Load Sample:", list(active_sample_map.keys()))
    selected_sample_path = active_sample_map[selected_sample_key]


# ─── Helper Functions ────────────────────────────────────────────────────────
def detect_modality(filename: str) -> str:
    """Detects media modality from file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    elif ext in AUDIO_EXTS:
        return "audio"
    elif ext in PDF_EXTS:
        return "pdf"
    return "unknown"


def render_modality_badge(modality: str, p4_on: bool = False):
    """Renders a clean inline badge showing detected file modality."""
    meta = MODALITY_META.get(modality, {})
    icon = meta.get("icon", "❓")
    label = meta.get("label", "Unknown")
    color = meta.get("color", "#8a99ad")
    pillars_str = meta.get("pillars", "—")

    if modality == "image" and p4_on:
        pillars_str += " + Pillar 4 (Benford OCR, if numeric digits detected)"

    st.markdown(f"""
    <div class="modality-badge" style="border-color: {color}40;">
        <span class="modality-icon">{icon}</span>
        <span style="color:{color}; font-weight:800; letter-spacing:0.5px;">
            AUTO-DETECTED MODALITY: {label.upper()}
        </span>
    </div>
    <div class="info-row">
        <span style="font-size:0.8rem; color:#8a99ad;">Active Forensic Engines →</span>
        <span style="font-size:0.8rem; color:#d1d5db;">{pillars_str}</span>
    </div>
    """, unsafe_allow_html=True)


def run_video_pipeline(video_path: str, original_filename: str, video_id: str):
    """Executes the full 10-step Pillar 2 Video Authenticity & Deepfake Forensics Pipeline."""
    if not PILLAR2_AVAILABLE:
        st.error("❌ Pillar 2 video backend dependencies could not be imported.")
        return

    st.markdown("#### 📽️ Video Playback Preview")
    st.video(video_path)

    run_btn = st.button("🚀 Run Deep Multi-Pillar Video Analysis", type="primary", use_container_width=True)
    stored_report = st.session_state.get(f"p2_result_{video_id}") or load_result(video_id)

    if run_btn:
        prog_bar = st.progress(5)
        status_text = st.empty()
        start_time = time.time()
        try:
            status_text.text("🔍 Step 1/10: Inspecting video container metadata & streams...")
            prog_bar.progress(10)
            v_details = inspect_video(video_path)

            status_text.text("🔬 Step 2/10: Running container forensic metadata analysis...")
            prog_bar.progress(20)
            meta_res = run_metadata_analysis(video_path, v_details)

            status_text.text("🎞️ Step 3/10: Extracting sampled keyframes...")
            prog_bar.progress(35)
            extracted_frames = extract_sampled_frames(video_path, video_id)

            status_text.text("👤 Step 4/10: Detecting human faces & facial landmarks...")
            prog_bar.progress(50)
            face_summary = detect_faces_in_frames(extracted_frames)

            status_text.text("🧠 Step 5/10: Evaluating visual ViT & boundary seam anomalies...")
            prog_bar.progress(65)
            v_res = run_visual_analysis(extracted_frames)

            status_text.text("📈 Step 6/10: Evaluating temporal continuity & flickering...")
            prog_bar.progress(75)
            t_res = run_temporal_analysis(extracted_frames)

            status_text.text("🎙️ Step 7/10: Extracting audio stream & analyzing acoustics...")
            prog_bar.progress(82)
            wav_path = extract_audio_track(video_path, video_id)
            a_res = run_audio_analysis(wav_path)

            status_text.text("👄 Step 8/10: Measuring mouth-to-speech lip synchronization...")
            prog_bar.progress(88)
            ls_res = run_lip_sync_analysis(extracted_frames, wav_path)

            status_text.text("⚖️ Step 9/10: Fusing multi-pillar features & classifying...")
            prog_bar.progress(94)
            classification = run_feature_fusion_and_classification(v_res, t_res, a_res, ls_res, meta_res)

            status_text.text("📌 Step 10/10: Extracting & annotating suspicious keyframes...")
            prog_bar.progress(98)
            suspicious_frames = extract_and_annotate_suspicious_frames(extracted_frames, video_id)

            proc_time = round(time.time() - start_time, 2)
            final_report = {
                "video_id": video_id,
                "file": {
                    "original_filename": original_filename,
                    "stored_filename": os.path.basename(video_path),
                    "format": os.path.splitext(video_path)[1].lstrip('.').lower(),
                    "size_mb": round(os.path.getsize(video_path)/(1024*1024), 2)
                },
                "video_details": v_details,
                "metadata": meta_res,
                "analysis": {
                    "face_detection": face_summary,
                    "visual": v_res,
                    "temporal": t_res,
                    "audio": a_res,
                    "lip_sync": ls_res
                },
                "classification": classification,
                "suspicious_frames": suspicious_frames,
                "processing": {
                    "status": "completed",
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_seconds": proc_time
                }
            }

            save_result(video_id, final_report)
            cleanup_temporary_frames(video_id)

            prog_bar.progress(100)
            status_text.success(f"✅ Video Analysis Completed in {proc_time}s! Result saved to JSON storage.")
            st.session_state[f"p2_result_{video_id}"] = final_report
            stored_report = final_report

        except Exception as e:
            prog_bar.progress(0)
            status_text.error(f"Video analysis failed: {e}")
            st.code(traceback.format_exc())

    # Display video analysis report
    if stored_report is not None:
        c_res = stored_report.get("classification", {})
        pred = c_res.get("prediction", "REAL")
        conf = c_res.get("confidence", 0.85) * 100
        scores = c_res.get("scores", {})

        if pred == "REAL":
            v_class = "unified-authentic"
            v_color = "#00f076"
            v_title = "AUTHENTIC VIDEO (NO MANIPULATION)"
        elif pred == "AI_GENERATED":
            v_class = "unified-ai-gen"
            v_color = "#ff3366"
            v_title = "AI GENERATED / DEEPFAKE ANOMALY"
        else:
            v_class = "unified-forged"
            v_color = "#f59e0b"
            v_title = "DIGITALLY FORGED / SPLICED MEDIA"

        st.markdown(f"""
        <div class="unified-verdict-card {v_class}">
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8a99ad;">
                    Pillar 2 Video Forensic Verdict
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {v_color}; margin-top: 4px;">
                    {v_title}
                </div>
                <div style="color: #8a99ad; font-size: 0.95rem; margin-top: 4px;">
                    Video ID: <b>{stored_report.get('video_id')}</b> • Target: <b>{stored_report.get('file', {}).get('original_filename')}</b>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: {v_color};">
                    {conf:.1f}%
                </div>
                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #8a99ad;">
                    Confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 3-Class Probability Distribution")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.metric("AI Generated", f"{scores.get('ai_generated', 0)*100:.1f}%")
        with p_col2:
            st.metric("Real / Genuine", f"{scores.get('real', 0)*100:.1f}%")
        with p_col3:
            st.metric("Digitally Forged", f"{scores.get('forged', 0)*100:.1f}%")

        st.divider()

        st.markdown("#### 🔬 Pillar 2 Sub-Forensic Decomposition")
        an_res = stored_report.get("analysis", {})
        meta_rep = stored_report.get("metadata", {})

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            v_anom = an_res.get("visual", {})
            v_col = "#ff3366" if v_anom.get("status") == "suspicious" else "#00f076"
            st.markdown(f"""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 2.1 • VISUAL & GAN SEAMS</div>
                <h4 style="color:{v_col}; margin: 0 0 8px 0;">{v_anom.get('status', 'normal').upper()}</h4>
                <div class='metric-chip'>Anomaly Score: <b>{v_anom.get('score', 0)*100:.1f}%</b></div>
                <div class='metric-chip'>Anomalies: <b>{', '.join(v_anom.get('anomalies', [])) or 'None'}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            t_anom = an_res.get("temporal", {})
            t_col = "#ff3366" if t_anom.get("status") == "suspicious" else "#00f076"
            st.markdown(f"""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 2.2 • TEMPORAL FLICKERING</div>
                <h4 style="color:{t_col}; margin: 0 0 8px 0;">{t_anom.get('status', 'normal').upper()}</h4>
                <div class='metric-chip'>Anomaly Score: <b>{t_anom.get('score', 0)*100:.1f}%</b></div>
                <div class='metric-chip'>Flickering: <b>{'DETECTED' if t_anom.get('flickering_detected') else 'CLEAR'}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            aud_anom = an_res.get("audio", {})
            aud_col = "#ff3366" if aud_anom.get("status") == "suspicious" else "#00f076"
            st.markdown(f"""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 2.3 • ACOUSTIC FORENSICS</div>
                <h4 style="color:{aud_col}; margin: 0 0 8px 0;">{aud_anom.get('status', 'normal').upper()}</h4>
                <div class='metric-chip'>Audio Track: <b>{'Available' if aud_anom.get('available') else 'No Audio'}</b></div>
                <div class='metric-chip'>Anomaly Score: <b>{aud_anom.get('score', 0)*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        col_d, col_e = st.columns(2)
        with col_d:
            ls_anom = an_res.get("lip_sync", {})
            ls_col = "#ff3366" if ls_anom.get("status") == "suspicious" else "#00f076"
            st.markdown(f"""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 2.4 • LIP-SYNC COHERENCE</div>
                <h4 style="color:{ls_col}; margin: 0 0 8px 0;">{ls_anom.get('status', 'normal').upper()}</h4>
                <div class='metric-chip'>Speech-Mouth Sync: <b>{ls_anom.get('status', 'N/A')}</b></div>
                <div class='metric-chip'>Correlation: <b>{ls_anom.get('correlation', 'N/A')}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_e:
            m_stat = meta_rep.get("metadata_status", "normal")
            m_col = "#ff3366" if m_stat == "suspicious" else "#00f076"
            st.markdown(f"""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 2.5 • CONTAINER & METADATA</div>
                <h4 style="color:{m_col}; margin: 0 0 8px 0;">{m_stat.upper()}</h4>
                <div class='metric-chip'>Codec: <b>{meta_rep.get('codec', 'Unknown')}</b></div>
                <div class='metric-chip'>Encoder: <b>{meta_rep.get('encoder', 'Unknown')}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Suspicious Keyframes
        st.markdown("#### 🚨 Flagged Suspicious Keyframes & Anomaly Markers")
        s_frames = stored_report.get("suspicious_frames", [])
        if s_frames:
            s_cols = st.columns(min(len(s_frames), 4))
            for idx, sf in enumerate(s_frames[:4]):
                with s_cols[idx]:
                    img_rel = sf.get("image", "")
                    img_full = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector", "storage", img_rel)
                    if os.path.exists(img_full):
                        st.image(img_full, caption=f"Frame #{sf.get('frame_number')} ({sf.get('timestamp_seconds')}s)", use_container_width=True)
                    st.markdown(f"**Anomaly:** {sf.get('reason')}")
                    st.markdown(f"**Score:** `{sf.get('score', 0)*100:.0f}%`")
        else:
            st.info("No suspicious keyframes exceeded anomaly threshold.")

        # JSON Telemetry Report
        st.divider()
        st.markdown("#### 📄 Zero-Database JSON Report")
        json_str = json.dumps(stored_report, indent=2)
        st.download_button(
            label="💾 Download Forensic JSON Verdict",
            data=json_str,
            file_name=f"{stored_report.get('video_id')}_verdict.json",
            mime="application/json"
        )
        with st.expander("🔍 View Raw JSON Telemetry"):
            st.code(json_str, language="json")

    # History Table
    st.divider()
    with st.expander("📜 View Past Video Analyses (Scanned from storage/results/*.json)"):
        if PILLAR2_AVAILABLE:
            hist_list = list_results()
            if hist_list:
                hist_df = pd.DataFrame(hist_list)
                disp_cols = [c for c in ['video_id', 'filename', 'prediction', 'confidence', 'date', 'size_mb'] if c in hist_df.columns]
                st.dataframe(hist_df[disp_cols], use_container_width=True)
            else:
                st.info("No past JSON analysis records found.")
        else:
            st.info("Pillar 2 storage services not available.")


# =============================================================================
# MODE 1: UNIVERSAL MULTI-PILLAR MEDIA ANALYSIS (AUTO-ROUTING)
# =============================================================================
if analysis_mode == "🌐 Universal Multi-Pillar Media Analysis":
    st.markdown("## 📤 Universal Forensic Media Ingestion")
    st.markdown(
        "Upload **any media file** (image, video, voice clip, or PDF document). "
        "The system automatically identifies the modality and executes all corresponding forensic pillars."
    )

    uploaded_file = st.file_uploader(
        "Upload Media File (Image · Video · Audio · PDF)",
        type=ALL_EXTS,
        label_visibility="collapsed",
    )
    st.markdown(
        "<p class='upload-hint'>Supported Formats: JPG · PNG · WEBP · BMP · MP4 · MOV · AVI · WAV · MP3 · FLAC · PDF</p>",
        unsafe_allow_html=True,
    )

    # Determine input bytes and file source
    source_file_bytes = None
    source_filename = ""

    if uploaded_file is not None:
        source_file_bytes = uploaded_file.read()
        source_filename = uploaded_file.name
    elif selected_sample_path and os.path.exists(selected_sample_path):
        with open(selected_sample_path, "rb") as f:
            source_file_bytes = f.read()
        source_filename = os.path.basename(selected_sample_path)

    if source_file_bytes is not None:
        modality = detect_modality(source_filename)
        st.divider()
        render_modality_badge(modality, p4_enabled)

        # ── ROUTE 1: IMAGE (Pillar 1 + Pillar 5 + Pillar 4 + Consensus) ──────
        if modality == "image":
            image_to_process = Image.open(io.BytesIO(source_file_bytes)).convert("RGB")
            img_np = np.array(image_to_process)

            with st.spinner("🔬 Running Visual Deepfake Forensics: ViT Neural Spectra + Shadow RANSAC Physics…"):
                p5_res = run_pillar5_inference(img_np, pil_img=image_to_process, p5_bundle=p5_bundle)
                p1_res = run_pillar1_inference(
                    image_to_process, processor_or_tfm, p1_model, device,
                    model_name=p1_model_name
                )
                p4_res = (
                    run_pillar4_inference(img_np, is_pdf=False)
                    if p4_enabled
                    else {
                        "applicable": False, "verdict": "DISABLED", "confidence": 0,
                        "digits_count": 0, "reason": "Pillar 4 disabled via sidebar toggle."
                    }
                )
                fusion_res = fuse_multi_pillar_verdict(
                    image_to_process, img_np, p1_res, p4_res, p5_res
                )

            # Master Consensus Verdict Banner
            render_verdict_banner(
                title="Consolidated Multi-Pillar Ensemble Verdict",
                verdict=fusion_res["verdict"],
                confidence=fusion_res["confidence"],
                target_name=source_filename,
                subtitle=f"• Engines: {p1_model_name} + {p5_model_filename}",
                is_real=fusion_res["is_real"],
                override_reason=fusion_res["override_reason"],
            )

            # 4-Pillar Grid
            st.markdown("### 🔍 Forensic Pillar Breakdown")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    '<div class="pillar-card"><div class="pillar-tag">PILLAR 1 • VISION TRANSFORMER</div>'
                    '<h3 style="margin:0 0 10px 0;">ViT Neural Head</h3>',
                    unsafe_allow_html=True,
                )
                p1_color = "#00f076" if p1_res["verdict"] == "AUTHENTIC" else "#ff3366"
                st.markdown(f"<h4 style='color:{p1_color};margin:0;'>{p1_res['verdict']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-chip'>Confidence: <b>{p1_res['confidence']:.2f}%</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-chip'>Model: <b>{p1_model_name}</b></div></div>", unsafe_allow_html=True)

            with col2:
                st.markdown(
                    '<div class="pillar-card pillar-card--dim"><div class="pillar-tag" style="color:#8a99ad;">PILLAR 3 • ACOUSTIC</div>'
                    '<h3 style="margin:0 0 10px 0;">Voice / Audio Engine</h3>',
                    unsafe_allow_html=True,
                )
                st.markdown("<h4 style='color:#8a99ad;margin:0;'>N/A (IMAGE)</h4>", unsafe_allow_html=True)
                st.markdown("<div class='metric-chip' style='color:#8a99ad;'>Upload audio/video for acoustic analysis</div></div>", unsafe_allow_html=True)

            with col3:
                p4_header = '<div class="pillar-card"><div class="pillar-tag">PILLAR 4 • DOCUMENT OCR</div><h3 style="margin:0 0 10px 0;">Benford\'s Law</h3>'
                if not p4_enabled:
                    st.markdown(p4_header, unsafe_allow_html=True)
                    st.markdown("<h4 style='color:#ffaa00;margin:0;'>DISABLED</h4>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-chip' style='color:#ffaa00;'>Toggle ON in sidebar</div></div>", unsafe_allow_html=True)
                elif p4_res.get("applicable"):
                    p4_color = "#00f076" if "AUTHENTIC" in p4_res["verdict"] else "#ff3366"
                    st.markdown(p4_header, unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color:{p4_color};margin:0;'>{p4_res['verdict']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-chip'>Confidence: <b>{p4_res['confidence']}%</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-chip'>Digits: <b>{p4_res['digits_count']}</b></div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(p4_header, unsafe_allow_html=True)
                    st.markdown("<h4 style='color:#8a99ad;margin:0;'>N/A (NATURAL SCENE)</h4>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-chip' style='color:#8a99ad;'>No numeric digits found</div></div>", unsafe_allow_html=True)

            with col4:
                st.markdown(
                    '<div class="pillar-card"><div class="pillar-tag">PILLAR 5 • PHYSICAL GEOMETRY</div>'
                    '<h3 style="margin:0 0 10px 0;">Shadow RANSAC</h3>',
                    unsafe_allow_html=True,
                )
                p5_color = "#00f076" if "AUTHENTIC" in p5_res["verdict"] else "#ff3366"
                st.markdown(f"<h4 style='color:{p5_color};margin:0;'>{p5_res['verdict']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-chip'>Confidence: <b>{p5_res['confidence']}%</b></div>", unsafe_allow_html=True)
                srm = p5_res.get('srm_var_4', 0.0)
                anomaly_label = 'Anomaly' if p5_res.get('is_prnu_anomaly') else 'Camera Sensor'
                st.markdown(f"<div class='metric-chip'>SRM4: <b>{srm:.1f}</b> ({anomaly_label})</div></div>", unsafe_allow_html=True)

            # Visual Evidence & RANSAC Geometry
            st.divider()
            st.markdown("### 🖼️ Visual Evidence & Forensic Ray Vectors")
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.image(image_to_process, caption=f"Original Uploaded Target: {source_filename}", use_container_width=True)
            with vcol2:
                if p5_res.get("overlay") is not None:
                    st.image(p5_res["overlay"], caption="Pillar 5: RANSAC Vanishing Point & Shadow Convergence Vectors", use_container_width=True)
                else:
                    st.info("No shadow vectors detected for visual overlay.")

            # Benford Chart if applicable
            if p4_enabled and p4_res.get("applicable"):
                st.divider()
                st.markdown("### 📊 Pillar 4 — Benford's Law Digit Distribution (Image OCR)")
                bcol1, bcol2 = st.columns([1, 1])
                with bcol1:
                    fig = plot_benford_distribution(p4_res["obs_freqs"], p4_res["expected_freqs"], p4_res["is_authentic"])
                    st.pyplot(fig)
                with bcol2:
                    st.markdown("#### 📄 OCR Extraction Preview")
                    if p4_res.get("extracted_image") is not None:
                        st.image(p4_res["extracted_image"], caption="Extracted Preprocessed ROI", use_container_width=True)
                    if p4_res.get("sample_text"):
                        with st.expander("📝 Extracted OCR Text"):
                            st.code(p4_res["sample_text"])

        # ── ROUTE 2: VIDEO (Pillar 2 Deepfake Pipeline) ──────────────────────
        elif modality == "video":
            temp_vid_dir = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector", "storage", "uploads")
            os.makedirs(temp_vid_dir, exist_ok=True)
            v_id = f"VID_{uuid.uuid4().hex[:6].upper()}"
            ext = os.path.splitext(source_filename)[1].lower() or ".mp4"
            v_save_path = os.path.join(temp_vid_dir, f"{v_id}{ext}")
            with open(v_save_path, "wb") as f:
                f.write(source_file_bytes)

            run_video_pipeline(v_save_path, source_filename, v_id)

        # ── ROUTE 3: AUDIO (Pillar 3 Acoustic & Speech Forensics) ────────────
        elif modality == "audio":
            st.markdown("### 🎙️ Pillar 3 — Acoustic & Synthetic Voice Forensics")
            ext = source_filename.rsplit(".", 1)[-1].lower() if "." in source_filename else "wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(source_file_bytes)
                audio_path = tmp.name

            st.markdown('<div class="audio-card">', unsafe_allow_html=True)
            st.audio(audio_path, format=f"audio/{ext}")
            st.markdown("</div>", unsafe_allow_html=True)

            internal_mode = "music" if "Music" in audio_submode else "spoken"

            with st.spinner("🎧 Executing Pillar 3 Acoustic Spectrogram & Waveform Forensics…"):
                try:
                    p3_res = classify_audio(audio_path, mode=internal_mode)
                    is_real = p3_res["prediction"] == "REAL"

                    render_verdict_banner(
                        title="Pillar 3 Acoustic Synthetic Voice Verdict",
                        verdict=f"{p3_res['prediction']} ({'AUTHENTIC HUMAN VOICE' if is_real else 'SYNTHETIC AI VOICE / CLONED'})",
                        confidence=p3_res["confidence"],
                        target_name=source_filename,
                        subtitle=f"• Model: {p3_res['model_used']}",
                        is_real=is_real,
                    )

                    acol1, acol2, acol3, acol4 = st.columns(4)
                    with acol1:
                        st.metric("Human Voice Probability", f"{p3_res['real_confidence']:.2f}%")
                    with acol2:
                        st.metric("AI Cloned Probability",   f"{p3_res['fake_confidence']:.2f}%")
                    with acol3:
                        st.metric("Duration / Sample Rate",  f"{p3_res['duration']:.1f}s @ {p3_res['samplerate']}Hz")
                    with acol4:
                        st.metric("Music / Beats Detected",  "Yes (HPSS Applied)" if p3_res["is_music"] else "No (Pure Speech)")

                    if p3_res.get("disclaimer"):
                        st.info(f"ℹ️ {p3_res['disclaimer']}")

                except Exception as e:
                    st.error(f"Error processing audio track: {e}")
                finally:
                    try:
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                    except Exception:
                        pass

        # ── ROUTE 4: PDF (Pillar 4 Document & Benford's Law OCR) ─────────────
        elif modality == "pdf":
            st.markdown("### 📄 Pillar 4 — Document, Invoice & PDF Statistical Forensics")
            st.markdown('<div class="benford-card">', unsafe_allow_html=True)
            st.markdown(f"**📁 Analyzing Document:** `{source_filename}`")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.spinner("📄 Performing PyMuPDF OCR Digit Extraction & Benford's Law Chi-Square Analysis…"):
                p4_doc_res = run_pillar4_inference(source_file_bytes, is_pdf=True)

            if p4_doc_res.get("applicable"):
                is_doc_auth = p4_doc_res["is_authentic"]
                render_verdict_banner(
                    title="Pillar 4 Document Forensic Verdict",
                    verdict=p4_doc_res["verdict"],
                    confidence=p4_doc_res["confidence"],
                    target_name=source_filename,
                    subtitle="• Benford First-Digit Chi² Goodness-of-Fit Test",
                    is_real=is_doc_auth,
                )

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Digits Extracted",        p4_doc_res["digits_count"])
                with mcol2:
                    st.metric("Mean Absolute Error (MAE)", f"{p4_doc_res['mae']:.4f}")
                with mcol3:
                    st.metric("Chi-Square (χ²)",          f"{p4_doc_res['chi_square']:.2f}")
                with mcol4:
                    st.metric("P-Value",                  f"{p4_doc_res['p_value']:.4f}")

                st.divider()
                v_col1, v_col2 = st.columns([1, 1])

                with v_col1:
                    st.markdown("#### 📊 Benford's Law Digit Distribution")
                    fig = plot_benford_distribution(
                        p4_doc_res["obs_freqs"], p4_doc_res["expected_freqs"], is_doc_auth
                    )
                    st.pyplot(fig)

                with v_col2:
                    st.markdown("#### 📄 Document Preview & OCR Stream")
                    if p4_doc_res.get("extracted_image") is not None:
                        st.image(p4_doc_res["extracted_image"], caption=f"Preview: {source_filename}", use_container_width=True)
                    if p4_doc_res.get("sample_text"):
                        with st.expander("📝 Extracted OCR Text"):
                            st.code(p4_doc_res["sample_text"])
            else:
                st.warning(f"Document analysis note: {p4_doc_res.get('reason', 'Could not process document.')}")

        else:
            st.error(f"❌ Unsupported file format: `{source_filename}`. Please upload an image, video, audio file, or PDF.")

    else:
        # Awaiting upload empty state
        st.markdown("""
        <div style="
            text-align:center;
            padding: 60px 20px;
            background: rgba(18,26,43,0.5);
            border: 2px dashed rgba(255,255,255,0.1);
            border-radius: 20px;
            margin-top: 10px;
        ">
            <div style="font-size:3.5rem; margin-bottom:12px;">🛡️</div>
            <h3 style="color:#8a99ad; font-weight:600; margin:0 0 10px 0;">
                Awaiting Universal Media Upload
            </h3>
            <p style="color:#8a99ad; font-size:0.95rem; max-width:620px; margin:0 auto 20px auto;">
                Drop a <b style="color:#4facfe;">photo</b> for visual deepfake analysis (Pillars 1 & 5),
                a <b style="color:#ec4899;">video clip</b> for deep video authenticity forensics (Pillar 2),
                an <b style="color:#a78bfa;">audio track</b> for synthetic speech detection (Pillar 3), or
                a <b style="color:#fb923c;">PDF document</b> for Benford's Law statistical forensics (Pillar 4).
            </p>
            <div class="info-row" style="justify-content:center;">
                <span class="pillar-chip">🖼️ Image → P1 + P5</span>
                <span class="pillar-chip">🎬 Video → P2 (All 5 Sub-Engines)</span>
                <span class="pillar-chip">🎧 Audio → P3</span>
                <span class="pillar-chip">📄 PDF → P4</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MODE 2: PILLAR 2: DEDICATED VIDEO AUTHENTICITY FORENSICS
# =============================================================================
elif analysis_mode == "🎬 Pillar 2: Video Authenticity & Deepfake Forensics":
    st.markdown("### 🎬 Pillar 2: Video Authenticity & Deepfake Forensics")
    st.markdown(
        "Inspects videos across **Visual Face/GAN Artifacts**, **Temporal Stability & Flickering**, "
        "**Audio Acoustics**, and **Lip-Sync Coherence**, backed by **Zero-Database JSON Storage**."
    )

    p2_col1, p2_col2 = st.columns([1.2, 0.8])
    with p2_col1:
        uploaded_video = st.file_uploader(
            "📤 Upload Video to Analyze (MP4, MOV, AVI, MKV, WEBM)...",
            type=["mp4", "mov", "avi", "mkv", "webm"]
        )
    with p2_col2:
        st.markdown("**Quick Load Test Video Samples:**")
        p2_sample_key = st.selectbox(
            "Select Pre-Configured Sample:",
            list(VIDEO_SAMPLES.keys())
        )
        p2_sample_path = VIDEO_SAMPLES.get(p2_sample_key)

    video_source_path = None
    original_video_name = ""
    video_id = f"VID_{uuid.uuid4().hex[:6].upper()}"

    if uploaded_video is not None:
        temp_vid_dir = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector", "storage", "uploads")
        os.makedirs(temp_vid_dir, exist_ok=True)
        ext = os.path.splitext(uploaded_video.name)[1].lower()
        save_path = os.path.join(temp_vid_dir, f"{video_id}{ext}")
        with open(save_path, "wb") as f:
            f.write(uploaded_video.read())
        video_source_path = save_path
        original_video_name = uploaded_video.name
    elif p2_sample_path and os.path.exists(p2_sample_path):
        video_source_path = p2_sample_path
        original_video_name = os.path.basename(p2_sample_path)
        video_id = os.path.splitext(original_video_name)[0]

    if video_source_path and os.path.exists(video_source_path):
        run_video_pipeline(video_source_path, original_video_name, video_id)
    else:
        st.info("👆 Please upload a video file or select a pre-loaded sample from above to begin Pillar 2 forensics.")


# =============================================================================
# MODE 3: PILLAR 3: DEDICATED VOICE & AUDIO SYNTHETIC SPEECH FORENSICS
# =============================================================================
elif analysis_mode == "🎙️ Pillar 3: Voice & Audio Synthetic Speech Forensics":
    st.markdown("### 🎙️ Pillar 3: Voice & Audio Synthetic Speech Forensics")
    st.markdown(
        "Analyzes conversational speech, voice notes, and musical tracks using "
        "**HuggingFace Wav2Vec2 + Librosa HPSS Harmonic-Percussive Separation**."
    )

    p3_col1, p3_col2 = st.columns([1.2, 0.8])
    with p3_col1:
        uploaded_audio = st.file_uploader(
            "📤 Upload Audio Track (WAV, MP3, FLAC, OGG, M4A)...",
            type=["wav", "mp3", "flac", "ogg", "m4a"]
        )
    with p3_col2:
        st.markdown("**Quick Load Test Audio Samples:**")
        p3_sample_key = st.selectbox("Select Audio Sample:", list(AUDIO_SAMPLES.keys()))
        p3_sample_path = AUDIO_SAMPLES.get(p3_sample_key)

    audio_bytes = None
    audio_fname = ""

    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        audio_fname = uploaded_audio.name
    elif p3_sample_path and os.path.exists(p3_sample_path):
        with open(p3_sample_path, "rb") as f:
            audio_bytes = f.read()
        audio_fname = os.path.basename(p3_sample_path)

    if audio_bytes is not None:
        ext = audio_fname.rsplit(".", 1)[-1].lower() if "." in audio_fname else "wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        st.markdown('<div class="audio-card">', unsafe_allow_html=True)
        st.audio(audio_path, format=f"audio/{ext}")
        st.markdown("</div>", unsafe_allow_html=True)

        internal_mode = "music" if "Music" in audio_submode else "spoken"

        with st.spinner("🎧 Executing Pillar 3 Acoustic Spectrogram & Waveform Forensics…"):
            try:
                p3_res = classify_audio(audio_path, mode=internal_mode)
                is_real = p3_res["prediction"] == "REAL"

                render_verdict_banner(
                    title="Pillar 3 Acoustic Synthetic Voice Verdict",
                    verdict=f"{p3_res['prediction']} ({'AUTHENTIC HUMAN VOICE' if is_real else 'SYNTHETIC AI VOICE / CLONED'})",
                    confidence=p3_res["confidence"],
                    target_name=audio_fname,
                    subtitle=f"• Model: {p3_res['model_used']}",
                    is_real=is_real,
                )

                acol1, acol2, acol3, acol4 = st.columns(4)
                with acol1:
                    st.metric("Human Voice Probability", f"{p3_res['real_confidence']:.2f}%")
                with acol2:
                    st.metric("AI Cloned Probability",   f"{p3_res['fake_confidence']:.2f}%")
                with acol3:
                    st.metric("Duration / Sample Rate",  f"{p3_res['duration']:.1f}s @ {p3_res['samplerate']}Hz")
                with acol4:
                    st.metric("Music / Beats Detected",  "Yes (HPSS Applied)" if p3_res["is_music"] else "No (Pure Speech)")

                if p3_res.get("disclaimer"):
                    st.info(f"ℹ️ {p3_res['disclaimer']}")

            except Exception as e:
                st.error(f"Error processing audio track: {e}")
            finally:
                try:
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                except Exception:
                    pass
    else:
        st.info("👆 Please upload an audio file (.wav, .mp3) or choose a test sample from above to inspect Pillar 3 synthetic speech forensics.")


# =============================================================================
# MODE 4: PILLAR 4: DEDICATED DOCUMENT, INVOICE & PDF STATISTICAL FORENSICS
# =============================================================================
elif analysis_mode == "📄 Pillar 4: Document, Invoice & PDF Statistical Forensics":
    st.markdown("### 📄 Pillar 4: Document, Invoice & PDF Statistical Forensics")
    st.markdown(
        "Analyzes OCR digit distributions and verifies adherence to **Benford's Law** "
        "to uncover forged or AI-synthesized financial records, receipts, and invoices."
    )

    p4_col1, p4_col2 = st.columns([1.2, 0.8])
    with p4_col1:
        uploaded_doc = st.file_uploader(
            "📤 Upload Document or PDF (PDF, PNG, JPG, JPEG, TIFF)...",
            type=["pdf", "png", "jpg", "jpeg", "tiff"]
        )
    with p4_col2:
        st.markdown("**Quick Load Test Document Samples:**")
        p4_sample_key = st.selectbox("Select Invoice Sample:", list(DOC_SAMPLES.keys()))
        p4_sample_path = DOC_SAMPLES.get(p4_sample_key)

    doc_bytes = None
    doc_name = ""
    is_pdf_file = False

    if uploaded_doc is not None:
        doc_bytes = uploaded_doc.read()
        doc_name = uploaded_doc.name
        is_pdf_file = doc_name.lower().endswith(".pdf")
    elif p4_sample_path and os.path.exists(p4_sample_path):
        with open(p4_sample_path, "rb") as f:
            doc_bytes = f.read()
        doc_name = os.path.basename(p4_sample_path)
        is_pdf_file = doc_name.lower().endswith(".pdf")

    if doc_bytes is not None:
        st.markdown('<div class="benford-card">', unsafe_allow_html=True)
        st.markdown(f"**📁 Analyzing Document:** `{doc_name}`")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("📄 Performing OCR Digit Extraction & Benford's Law Chi-Square Analysis…"):
            p4_doc_res = run_pillar4_inference(doc_bytes, is_pdf=is_pdf_file)

        if p4_doc_res.get("applicable"):
            is_doc_auth = p4_doc_res["is_authentic"]
            render_verdict_banner(
                title="Pillar 4 Document Forensic Verdict",
                verdict=p4_doc_res["verdict"],
                confidence=p4_doc_res["confidence"],
                target_name=doc_name,
                subtitle="• Benford First-Digit Chi² Goodness-of-Fit Test",
                is_real=is_doc_auth,
            )

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Digits Extracted",        p4_doc_res["digits_count"])
            with mcol2:
                st.metric("Mean Absolute Error (MAE)", f"{p4_doc_res['mae']:.4f}")
            with mcol3:
                st.metric("Chi-Square (χ²)",          f"{p4_doc_res['chi_square']:.2f}")
            with mcol4:
                st.metric("P-Value",                  f"{p4_doc_res['p_value']:.4f}")

            st.divider()
            v_col1, v_col2 = st.columns([1, 1])

            with v_col1:
                st.markdown("#### 📊 Benford's Law Digit Distribution")
                fig = plot_benford_distribution(
                    p4_doc_res["obs_freqs"], p4_doc_res["expected_freqs"], is_doc_auth
                )
                st.pyplot(fig)

            with v_col2:
                st.markdown("#### 📄 Document Preview & OCR Text Stream")
                if p4_doc_res.get("extracted_image") is not None:
                    st.image(
                        p4_doc_res["extracted_image"],
                        caption=f"Preview: {doc_name}",
                        use_container_width=True,
                    )
                if p4_doc_res.get("sample_text"):
                    with st.expander("📝 Extracted OCR Text Sample"):
                        st.code(p4_doc_res["sample_text"])
        else:
            st.warning(f"Document analysis note: {p4_doc_res.get('reason', 'Could not process document.')}")
    else:
        st.info("👆 Please upload a PDF or invoice image from above to execute Pillar 4 document forensics.")
