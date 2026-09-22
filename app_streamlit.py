"""
================================================================================
Universal Synthetic Media Forensics Engine (USMFE) - Streamlit Web Dashboard
================================================================================
All-in-One Universal Upload: auto-detects file modality and routes to the
correct forensic pillars automatically.
  • Image  (JPG, PNG, WEBP, BMP, TIFF) → Pillar 1 (ViT) + Pillar 5 (Shadow RANSAC)
                                          + Pillar 4 (Benford OCR) — if P4 toggle ON
  • Audio  (WAV, MP3, FLAC, OGG, M4A)  → Pillar 3 (Acoustic Transformer + HPSS)
  • PDF    (PDF)                         → Pillar 4 (Benford's Law OCR Statistical)
"""

import io
import os
import tempfile
import numpy as np
from PIL import Image
import streamlit as st

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_EXTS  = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
AUDIO_EXTS  = {"wav", "mp3", "flac", "ogg", "m4a"}
PDF_EXTS    = {"pdf"}
ALL_EXTS    = list(IMAGE_EXTS | AUDIO_EXTS | PDF_EXTS)

MODALITY_META = {
    "image": {
        "icon": "🖼️",
        "label": "Visual Photo / Render",
        "color": "#4facfe",
        "pillars": "Pillar 1 (ViT Neural Forensics) + Pillar 5 (Shadow RANSAC Physics)",
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

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Pillar Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()

# ─── Additional CSS ───────────────────────────────────────────────────────────
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
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ffaa00;
        margin-bottom: 4px;
    }
    /* Upload zone hint */
    .upload-hint {
        text-align: center;
        color: #8a99ad;
        font-size: 0.92rem;
        margin-top: 6px;
        font-style: italic;
    }
    /* Section divider label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8a99ad;
        margin-bottom: 10px;
    }
    /* Audio wave card */
    .audio-card {
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }
    /* Benford card */
    .benford-card {
        background: rgba(251,146,60,0.07);
        border: 1px solid rgba(251,146,60,0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }
    /* Info row */
    .info-row {
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

render_header()

# ─── Load Core Models ─────────────────────────────────────────────────────────
processor_or_tfm, p1_model, device, p1_model_name = load_pillar1_vit()
p5_bundle, p5_model_filename = load_pillar5_ml_bundle()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Forensic Controls")

    # ── Pillar 4 Toggle ───────────────────────────────────────────────────────
    st.markdown("""<div class="toggle-card">
        <div class="toggle-label">🔬 Pillar 4 — Benford's Law OCR</div>
    </div>""", unsafe_allow_html=True)

    p4_enabled = st.toggle(
        "Enable Document / Invoice Analysis",
        value=True,
        help=(
            "When ON: images that contain numeric/tabular content are also analyzed "
            "using Benford's Law OCR digit-distribution chi-square test. "
            "PDFs are always routed to Pillar 4 regardless of this toggle."
        )
    )
    st.caption(
        "✅ Active — applies to images with numeric content" if p4_enabled
        else "⛔ Disabled — image analysis skips OCR Benford check"
    )
    st.divider()

    # ── Audio Sub-Mode ────────────────────────────────────────────────────────
    st.markdown("**🎙️ Audio Analysis Sub-Mode**")
    audio_submode = st.radio(
        "Voice / Music pipeline:",
        ["🗣️ Spoken Voice (Standard)", "🎵 Music Track (Experimental / Demixed)"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()

    # ── Engine Status ─────────────────────────────────────────────────────────
    st.markdown("### 🏛️ Forensic Engine Status")
    st.markdown(f"🔹 **P1 ViT:** `{p1_model_name}`")
    st.markdown("🔹 **P3 Audio:** HuggingFace Wav2Vec2 + HPSS")
    p4_status = "🟢 Active" if p4_enabled else "🔴 Disabled"
    st.markdown(f"🔹 **P4 Benford OCR:** {p4_status}")
    st.markdown(f"🔹 **P5 Ensemble:** `{p5_model_filename}` (Authoritative)")
    st.divider()

    # ── Pre-loaded Samples ────────────────────────────────────────────────────
    st.markdown("### 🧪 Pre-loaded Test Samples")
    sample_category = st.selectbox("Category", ["Images", "Audio", "Documents"])

    IMAGE_SAMPLES = {
        "Select…": None,
        "AI_img1.jpeg (Fake AI)":    os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img1.jpeg"),
        "AI_img2.jpeg (Fake AI)":    os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img2.jpeg"),
        "AI_img3.png (Fake AI)":     os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img3.png"),
        "AI_img4.png (Fake AI)":     os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img4.png"),
        "Real_img1.JPG (Real Photo)": os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img1.JPG"),
        "Real_img2.jpg (Real Photo)": os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img2.jpg"),
        "Real_img3.jpg (Real Photo)": os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img3.jpg"),
    }
    AUDIO_SAMPLES = {
        "Select…": None,
        "sample_test.wav": os.path.join(BASE_DIR, "test_audio", "sample_test.wav"),
    }
    DOC_SAMPLES = {
        "Select…": None,
        "invoice_1.png":   os.path.join(BASE_DIR, "Pillar 4", "invoice_1.png"),
        "invoice_01.png":  os.path.join(BASE_DIR, "Pillar 4", "invoice_01.png"),
        "invoice_2.jpg":   os.path.join(BASE_DIR, "Pillar 4", "invoice_2.jpg"),
        "invoice_3.png":   os.path.join(BASE_DIR, "Pillar 4", "invoice_3.png"),
    }

    if sample_category == "Images":
        sample_map = IMAGE_SAMPLES
    elif sample_category == "Audio":
        sample_map = AUDIO_SAMPLES
    else:
        sample_map = DOC_SAMPLES

    selected_sample_key = st.selectbox("Select sample:", list(sample_map.keys()))
    selected_sample_path = sample_map[selected_sample_key]

# ─── Helper: detect modality from filename ────────────────────────────────────
def detect_modality(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in IMAGE_EXTS:
        return "image"
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

    # If image + p4 ON mention it
    if modality == "image" and p4_on:
        pillars_str += " + Pillar 4 (Benford OCR, if numeric digits detected)"

    st.markdown(f"""
    <div class="modality-badge" style="border-color: {color}40;">
        <span class="modality-icon">{icon}</span>
        <span style="color:{color}; font-weight:800; letter-spacing:0.5px;">
            AUTO-DETECTED: {label.upper()}
        </span>
    </div>
    <div class="info-row">
        <span style="font-size:0.8rem; color:#8a99ad;">Active Pillars →</span>
        <span style="font-size:0.8rem; color:#d1d5db;">{pillars_str}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN UPLOAD ZONE ─────────────────────────────────────────────────────────
st.markdown("## 📤 Universal Forensic File Upload")
st.markdown(
    "Drop **any** media file — photo, audio clip, or PDF — and the engine "
    "automatically selects the correct forensic pillars.",
    unsafe_allow_html=False,
)

uploaded_file = st.file_uploader(
    "Upload Media File (Image · Audio · PDF)",
    type=ALL_EXTS,
    label_visibility="collapsed",
)
st.markdown(
    "<p class='upload-hint'>Supports JPG · PNG · WEBP · BMP · TIFF · WAV · MP3 · FLAC · OGG · M4A · PDF</p>",
    unsafe_allow_html=True,
)

# ─── Resolve source (uploaded or pre-loaded sample) ───────────────────────────
source_file_bytes = None
source_filename   = ""

if uploaded_file is not None:
    source_file_bytes = uploaded_file.read()
    source_filename   = uploaded_file.name
elif selected_sample_path and os.path.exists(selected_sample_path):
    with open(selected_sample_path, "rb") as f:
        source_file_bytes = f.read()
    source_filename = os.path.basename(selected_sample_path)

# ─── Auto-route & run forensics ───────────────────────────────────────────────
if source_file_bytes is not None:
    modality = detect_modality(source_filename)

    st.divider()
    render_modality_badge(modality, p4_enabled)

    # ── IMAGE MODALITY: Pillar 1 + Pillar 5  (+Pillar 4 if toggle ON) ────────
    if modality == "image":
        image_to_process = Image.open(io.BytesIO(source_file_bytes)).convert("RGB")
        img_np = np.array(image_to_process)

        with st.spinner("🔬 Running Multi-Pillar Visual Deepfake Forensics Engine…"):
            p5_res = run_pillar5_inference(img_np, pil_img=image_to_process, p5_bundle=p5_bundle)
            p1_res = run_pillar1_inference(
                image_to_process, processor_or_tfm, p1_model, device,
                model_name=p1_model_name
            )
            p4_res = (
                run_pillar4_inference(img_np, is_pdf=False)
                if p4_enabled
                else {"applicable": False, "verdict": "DISABLED", "confidence": 0,
                      "digits_count": 0, "reason": "Pillar 4 disabled via toggle."}
            )
            fusion_res = fuse_multi_pillar_verdict(
                image_to_process, img_np, p1_res, p4_res, p5_res
            )

        # Master verdict
        render_verdict_banner(
            title="Consolidated Multi-Pillar Ensemble Verdict",
            verdict=fusion_res["verdict"],
            confidence=fusion_res["confidence"],
            target_name=source_filename,
            subtitle=f"• Engines: {p1_model_name} + {p5_model_filename}",
            is_real=fusion_res["is_real"],
            override_reason=fusion_res["override_reason"],
        )

        # 4-pillar breakdown grid
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
            # Pillar 3 — passive (not applicable for image)
            st.markdown(
                '<div class="pillar-card pillar-card--dim"><div class="pillar-tag" style="color:#8a99ad;">PILLAR 3 • ACOUSTIC</div>'
                '<h3 style="margin:0 0 10px 0;">Voice / Audio Engine</h3>',
                unsafe_allow_html=True,
            )
            st.markdown("<h4 style='color:#8a99ad;margin:0;'>N/A (IMAGE)</h4>", unsafe_allow_html=True)
            st.markdown("<div class='metric-chip' style='color:#8a99ad;'>Upload audio for voice analysis</div></div>", unsafe_allow_html=True)

        with col3:
            p4_header = '<div class="pillar-card"><div class="pillar-tag">PILLAR 4 • DOCUMENT OCR</div><h3 style="margin:0 0 10px 0;">Benford\'s Law</h3>'
            if not p4_enabled:
                st.markdown(p4_header, unsafe_allow_html=True)
                st.markdown("<h4 style='color:#ffaa00;margin:0;'>DISABLED (TOGGLE OFF)</h4>", unsafe_allow_html=True)
                st.markdown("<div class='metric-chip' style='color:#ffaa00;'>Enable via sidebar toggle</div></div>", unsafe_allow_html=True)
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

        # Visual evidence
        st.divider()
        st.markdown("### 🖼️ Visual Evidence & Forensic Ray Vectors")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.image(image_to_process, caption=f"Uploaded: {source_filename}", use_container_width=True)
        with vcol2:
            if p5_res.get("overlay") is not None:
                st.image(p5_res["overlay"], caption="Pillar 5: RANSAC Vanishing Point & Shadow Convergence Vectors", use_container_width=True)
            else:
                st.info("No shadow vectors detected for visual overlay.")

        # Benford chart (only if applicable & P4 enabled)
        if p4_enabled and p4_res.get("applicable"):
            st.divider()
            st.markdown("### 📊 Pillar 4 — Benford's Law Digit Distribution (Image OCR)")
            bcol1, bcol2 = st.columns([1, 1])
            with bcol1:
                fig = plot_benford_distribution(p4_res["obs_freqs"], p4_res["expected_freqs"], p4_res["is_authentic"])
                st.pyplot(fig)
            with bcol2:
                st.markdown("#### 📄 OCR Preview")
                if p4_res.get("extracted_image") is not None:
                    st.image(p4_res["extracted_image"], caption="Extracted Image for OCR", use_container_width=True)
                if p4_res.get("sample_text"):
                    with st.expander("📝 Extracted OCR Text"):
                        st.code(p4_res["sample_text"])

    # ── AUDIO MODALITY: Pillar 3 ──────────────────────────────────────────────
    elif modality == "audio":
        st.markdown("### 🎙️ Pillar 3 — Acoustic & Speech Synthetic Voice Forensics")
        st.markdown(
            "Analyzes speech and music using **Wav2Vec2 + Librosa HPSS** harmonic-percussive separation."
        )

        # Write to temp file
        ext = source_filename.rsplit(".", 1)[-1].lower() if "." in source_filename else "wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(source_file_bytes)
            audio_path = tmp.name

        st.markdown('<div class="audio-card">', unsafe_allow_html=True)
        st.audio(audio_path, format=f"audio/{ext}")
        st.markdown("</div>", unsafe_allow_html=True)

        internal_mode = "music" if "Music" in audio_submode or "Song" in audio_submode else "spoken"

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
        
        # Cleanup
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    # ── PDF MODALITY: Pillar 4 ────────────────────────────────────────────────
    elif modality == "pdf":
        st.markdown("### 📄 Pillar 4 — Document, Invoice & PDF Statistical Forensics")
        st.markdown(
            "Extracts numeric digit distributions and verifies adherence to **Benford's Law** "
            "to detect forged or AI-manipulated tabular records and invoices."
        )

        st.markdown('<div class="benford-card">', unsafe_allow_html=True)
        st.markdown(f"**📁 Analyzing PDF:** `{source_filename}`")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("📄 Performing OCR Digit Extraction & Benford's Law Chi-Square Analysis…"):
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
                st.markdown("#### 📄 Document Preview & OCR Text Stream")
                if p4_doc_res.get("extracted_image") is not None:
                    st.image(
                        p4_doc_res["extracted_image"],
                        caption=f"Preview: {source_filename}",
                        use_container_width=True,
                    )
                if p4_doc_res.get("sample_text"):
                    with st.expander("📝 Extracted OCR Text Sample"):
                        st.code(p4_doc_res["sample_text"])
        else:
            st.warning(f"Document analysis note: {p4_doc_res.get('reason', 'Could not process document.')}")

    # ── UNKNOWN ───────────────────────────────────────────────────────────────
    else:
        st.error(f"❌ Unsupported file type: `{source_filename}`. Please upload an image, audio file, or PDF.")

else:
    # ── Empty state ───────────────────────────────────────────────────────────
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
            Awaiting Media Upload
        </h3>
        <p style="color:#8a99ad; font-size:0.95rem; max-width:520px; margin:0 auto 20px auto;">
            Upload a <b style="color:#4facfe;">photo</b> for visual deepfake analysis (Pillars 1 & 5),
            an <b style="color:#a78bfa;">audio clip</b> for voice forensics (Pillar 3), or
            a <b style="color:#fb923c;">PDF</b> for document statistical forensics (Pillar 4).
        </p>
        <div class="info-row" style="justify-content:center;">
            <span class="pillar-chip">🖼️ Image → P1 + P5</span>
            <span class="pillar-chip">🎧 Audio → P3</span>
            <span class="pillar-chip">📄 PDF → P4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
