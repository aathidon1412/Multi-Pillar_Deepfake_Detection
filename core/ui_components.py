"""
================================================================================
UI Theme, Styles & Visual Assets for Multi-Pillar Deepfake Detection
================================================================================
Encapsulates CSS styles, Glassmorphism theme, and Matplotlib plotting helpers.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(127, 0, 255, 0.05) 0%, transparent 40%),
                    #0b0f19;
        color: #f0f4f8;
    }
    
    .header-box {
        background: rgba(18, 26, 43, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px 30px;
        backdrop-filter: blur(16px);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .project-badge {
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        color: #0b0f19;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff, #8a99ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .unified-verdict-card {
        padding: 25px;
        border-radius: 18px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .unified-authentic {
        background: linear-gradient(135deg, rgba(0, 240, 118, 0.15), rgba(0, 240, 118, 0.03));
        border: 2px solid #00f076;
    }
    
    .unified-fake {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.15), rgba(255, 51, 102, 0.03));
        border: 2px solid #ff3366;
    }
    
    .pillar-card {
        background: rgba(18, 26, 43, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        backdrop-filter: blur(12px);
    }
    
    .pillar-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #00f2fe;
        margin-bottom: 6px;
    }
    
    .metric-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        background: rgba(255, 255, 255, 0.04);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 6px;
    }
</style>
"""

def apply_custom_theme():
    """Injects custom cyber-forensics dark CSS theme."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_header():
    """Renders the top title banner."""
    st.markdown("""
    <div class="header-box">
        <div>
            <span class="project-badge">UNIVERSAL SYNTHETIC MEDIA FORENSICS ENGINE (USMFE)</span>
            <h1 class="main-title">Multi-Pillar Deepfake Detection</h1>
            <p style="color: #8a99ad; margin: 5px 0 0 0; font-size: 0.95rem;">
                Unified Cyber-Forensics Fusion: ViT Neural Spectra (P1) + Acoustic Voice Transformers (P3) + Document Semantics (P4) + Hybrid Shadow Physics ML (P5)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_verdict_banner(title: str, verdict: str, confidence: float, target_name: str, subtitle: str = "", is_real: bool = True, override_reason: str = None):
    """Renders a responsive, glowing master verdict card."""
    banner_class = "unified-authentic" if is_real else "unified-fake"
    verdict_color = "#00f076" if is_real else "#ff3366"
    override_html = f"<br><span style='color: #ffaa00; font-weight: 600;'>⚡ Forensic Override Active: {override_reason}</span>" if override_reason else ""
    
    st.markdown(f"""
    <div class="unified-verdict-card {banner_class}">
        <div>
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8a99ad;">
                {title}
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {verdict_color}; margin-top: 4px;">
                {verdict}
            </div>
            <div style="color: #8a99ad; font-size: 0.95rem; margin-top: 4px;">
                Target: <b>{target_name}</b> {subtitle}
                {override_html}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: {verdict_color};">
                {confidence:.1f}%
            </div>
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #8a99ad;">
                Confidence
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def plot_benford_distribution(obs_freqs, exp_freqs, is_authentic: bool):
    """Generates a dark-themed Benford's Law comparison chart."""
    digits_arr = np.arange(1, 10)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0e1626')
    ax.set_facecolor('#0e1626')
    
    bar_color = '#00f076' if is_authentic else '#ff3366'
    ax.bar(digits_arr, obs_freqs, color=bar_color, alpha=0.75, width=0.5, label='Observed Frequencies (%)')
    ax.plot(digits_arr, exp_freqs, color='#00f2fe', marker='o', linewidth=2.5, label="Benford's Law (Expected)")
    
    ax.set_xticks(digits_arr)
    ax.set_xlabel("Leading Digit (1-9)", color='#f0f4f8', fontweight='bold')
    ax.set_ylabel("Relative Frequency (%)", color='#f0f4f8', fontweight='bold')
    ax.legend(facecolor='#182234', edgecolor='none', labelcolor='#f0f4f8')
    ax.grid(color='#ffffff', alpha=0.1, linestyle='--')
    
    for spine in ax.spines.values():
        spine.set_color('#ffffff')
        spine.set_alpha(0.1)
        
    plt.tight_layout()
    return fig


def render_xai_explanation_panel(xai_data: dict, title: str = "Explainable AI (XAI) Forensic Evidence Dossier"):
    """Renders a standardized XAI evidence panel in Streamlit."""
    if not xai_data or not isinstance(xai_data, dict):
        return

    available = xai_data.get("xai_available", True)
    pillar = xai_data.get("pillar", "Forensic Pillar")
    human_exp = xai_data.get("human_explanation", "")
    tech_exp = xai_data.get("technical_explanation", "")
    evidence = xai_data.get("evidence", [])
    limitations = xai_data.get("limitations", [])

    st.markdown(f"""
    <div style="background: rgba(14, 22, 38, 0.95); border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 12px; padding: 20px; margin: 18px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.3rem;">🧠</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #00f2fe; letter-spacing: 0.5px;">{title}</span>
            </div>
            <span style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; background: rgba(0, 242, 254, 0.12); color: #00f2fe; padding: 4px 10px; border-radius: 20px; border: 1px solid rgba(0,242,254,0.3);">
                {pillar}
            </span>
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.8rem; text-transform: uppercase; color: #8a99ad; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px;">Plain-Language Analytical Summary</div>
            <p style="font-size: 0.95rem; color: #f0f4f8; line-height: 1.5; margin: 0;">{human_exp}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if evidence:
        st.markdown("##### 🔬 Evidence & Feature Importance Breakdown")
        for ev in evidence:
            direction = ev.get("direction", "neutral")
            dir_color = "#ff3366" if direction == "supports_fake" else ("#00f076" if direction == "supports_real" else "#8a99ad")
            dir_label = "SUPPORTS SYNTHETIC / FAKE" if direction == "supports_fake" else ("SUPPORTS AUTHENTIC / REAL" if direction == "supports_real" else "NEUTRAL")
            importance_pct = int(ev.get("importance", 0.5) * 100)
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border-left: 4px solid {dir_color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; color: #d1d5db;">{ev.get('feature')}</span>
                    <span style="font-size: 0.75rem; font-weight: 700; color: {dir_color};">{dir_label} • {importance_pct}% Importance</span>
                </div>
                <div style="font-size: 0.82rem; color: #8a99ad; line-height: 1.4;">{ev.get('description')}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🔍 Researcher Deep Telemetry & Methodological Limitations"):
        st.markdown(f"**Technical Findings:**\n`{tech_exp}`")
        if limitations:
            st.markdown("**Methodological Considerations & Scope Limitations:**")
            for lim in limitations:
                st.markdown(f"- {lim}")


def render_pillar1_xai_inspector(xai_data: dict, orig_pil_img=None):
    """
    Renders the dedicated interactive Pillar 1 Vision Transformer XAI Inspector.
    Provides:
      - [Original Image] & [AI Explanation Heatmap]
      - Dynamic Heatmap Opacity Slider & Alpha Blending
      - Show/Hide explanation toggle
      - Target prediction & confidence
      - 'What does this mean?' expandable accordion
      - 'Technical details' expandable section
      - Legend: Low influence ───────── High influence
      - Forensic Disclaimer
      - Graceful error fallback
    """
    if not xai_data or not isinstance(xai_data, dict):
        return

    st.markdown("### 🧠 Explain Prediction (Pillar 1: Vision Transformer)")

    if not xai_data.get("xai_available", False):
        st.warning("⚠️ **Explanation unavailable for this sample.** The standard prediction remains valid.")
        return

    prediction = xai_data.get("prediction", "UNKNOWN")
    confidence = xai_data.get("confidence", 85.0)
    human_exp = xai_data.get("human_explanation", "")
    tech_exp = xai_data.get("technical_explanation", "")
    visualizations = xai_data.get("visualizations", [])
    vis = visualizations[0] if visualizations else {}

    show_exp = st.checkbox("Show Explanation Heatmap", value=True, key="p1_show_xai_toggle")
    if not show_exp:
        return

    # Attribution callout banner
    st.info(f"💡 **Attribution Summary:** {human_exp}")

    # Layout: Image & Heatmap with Interactive Opacity Slider
    opacity = st.slider("Heatmap Overlay Opacity", min_value=0.0, max_value=1.0, value=0.60, step=0.05, key="p1_opacity_slider")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📷 [Original Image]")
        if orig_pil_img is not None:
            st.image(orig_pil_img, use_container_width=True, caption=f"Target Media • Prediction: {prediction} ({confidence:.1f}%)")
        else:
            st.info("Original image input")

    with col2:
        st.markdown("##### 🔬 [AI Explanation Heatmap]")
        heatmap_url = vis.get("heatmap_url")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        heatmap_file_path = os.path.join(base_dir, "Pillar 2", "video-authenticity-detector", "storage", heatmap_url) if heatmap_url else None
        
        if heatmap_file_path and os.path.exists(heatmap_file_path):
            heatmap_img = Image.open(heatmap_file_path)
            if orig_pil_img is not None and opacity > 0:
                # Dynamically alpha-blend in memory for Streamlit
                orig_resized = orig_pil_img.convert("RGB").resize(heatmap_img.size)
                blended = Image.blend(orig_resized, heatmap_img.convert("RGB"), alpha=opacity)
                st.image(blended, use_container_width=True, caption=f"ViT Attention Rollout Overlay (Opacity: {int(opacity*100)}%)")
            else:
                st.image(heatmap_img, use_container_width=True, caption="ViT Token Attention Density Map (JET)")
        else:
            st.info("Heatmap visualization ready in data URI.")

    # Colorbar Legend
    st.markdown("""
    <div style="margin: 14px 0 18px 0; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 16px;">
        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #8a99ad; margin-bottom: 6px;">
            <span>Low influence</span>
            <span style="color: #00f2fe;">Attention Gradient Flow Scale</span>
            <span>High influence</span>
        </div>
        <div style="height: 10px; width: 100%; border-radius: 5px; background: linear-gradient(to right, #000080 0%, #0000ff 20%, #00ffff 40%, #00ff00 60%, #ffff00 80%, #ff0000 100%);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Accordions
    with st.expander("❓ What does this mean?"):
        st.markdown(f"""
        - The Vision Transformer divides the image into a 14×14 grid of 196 distinct patches.
        - **Attention Rollout** tracks how the neural network passes attention across all 12 transformer layers from the classification token (`[CLS]`) to individual spatial patches.
        - Highlighted regions (yellow/red) represent the visual image patches that received higher model attention for the **{prediction}** prediction.
        """)

    with st.expander("🔬 Technical details"):
        st.markdown(f"""
        - **Technical Finding:** `{tech_exp}`
        - **Peak Region:** `{vis.get('peak_region', 'central focal region')}`
        - **Peak Normalized Coordinates:** `X={vis.get('peak_coords', [0.5, 0.5])[0]}, Y={vis.get('peak_coords', [0.5, 0.5])[1]}`
        - **Method:** Attention Rollout (Abnar & Zuidema, 2020) across 12 Multi-Head Attention blocks
        """)

    # Mandatory Forensic Disclaimer
    st.markdown("""
    <div style="background: rgba(255, 170, 0, 0.08); border-left: 4px solid #ffaa00; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 0.85rem; color: #ffcc66;">
        ⚠️ <strong>Forensic Disclaimer:</strong> Highlighted regions indicate model attribution and should be interpreted as supporting evidence, not proof of manipulation.
    </div>
    """, unsafe_allow_html=True)


def render_pillar2_temporal_xai_section(xai_data: dict, video_duration: float = 0.0):
    """
    Renders the Explainable AI (XAI) Temporal Evidence Attribution section for Pillar 2 in Streamlit.
    Features:
      - Interactive Suspicious Segment Timeline with selectable markers
      - Anomaly details: Timestamp, Primary evidence, Supporting evidence
      - Signal Contribution horizontal bars with direction tags (supports_fake / supports_real)
      - Plain-English Human Explanation
      - Technical Telemetry & Forensic Disclaimer
    """
    if not xai_data or not isinstance(xai_data, dict):
        return

    st.markdown("### ⏱️ Why was this part of the video flagged? (Temporal Evidence Attribution)")

    if not xai_data.get("xai_available", False):
        st.warning("⚠️ **Explanation unavailable for this sample.** The standard video forensic prediction remains valid.")
        return

    segments = xai_data.get("temporal_segments", [])
    if not segments:
        st.info("ℹ️ **No significant temporal anomaly was identified by the available video-forensic signals.**\n\n*(Note: This absence alone does not establish absolute authenticity).*")
        return

    human_summary = xai_data.get("human_explanation", "")
    st.info(f"💡 **Forensic Evidence Summary:** {human_summary}")

    # Timeline Selector
    st.markdown("##### 📍 Detected Suspicious Segments")
    segment_options = []
    for i, seg in enumerate(segments):
        ts_str = seg.get('timestamp_display') or f"{seg.get('start_time', 0):.1f}s"
        sig_str = seg.get('primary_signal', 'Anomaly')
        sev_str = seg.get('severity', 'MED')
        segment_options.append(f"Segment #{i+1}: {ts_str} — {sig_str} [{sev_str}]")

    selected_idx = st.selectbox(
        "Select an evidence segment to inspect forensic attribution:",
        range(len(segments)),
        format_func=lambda idx: segment_options[idx],
        key="p2_timeline_selector"
    )

    sel_seg = segments[selected_idx]

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.metric("Timestamp Window", sel_seg.get("timestamp_display", f"{sel_seg.get('start_time', 0):.1f}s"))
    with col_meta2:
        st.metric("Primary Forensic Signal", sel_seg.get("primary_signal", "Visual"))
    with col_meta3:
        st.metric("Anomaly Severity", sel_seg.get("severity", "MEDIUM"), delta=f"Score: {sel_seg.get('score', 0):.2f}")

    # Human-Readable Explanation Callout
    st.markdown(f"""
    <div style="background: rgba(0, 242, 254, 0.05); border-left: 4px solid #00f2fe; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
        <div style="color: #00f2fe; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
            Human-Readable Segment Explanation
        </div>
        <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
            {sel_seg.get('human_explanation', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Signal Contributions Chart
    st.markdown("##### 📊 Forensic Signal Attribution & Contribution Weights")
    contributions = sel_seg.get("signal_contributions", [])
    for sc in contributions:
        sig_name = sc.get("signal", "")
        sig_score = float(sc.get("score", 0.0))
        sig_dir = sc.get("direction", "supports_real")
        sig_desc = sc.get("description", "")
        is_fake = (sig_dir == "supports_fake")
        bar_color = "#ff3366" if is_fake else "#00f076"
        badge_text = "SUPPORTS FAKE" if is_fake else "SUPPORTS REAL"
        badge_bg = "rgba(255, 51, 102, 0.15)" if is_fake else "rgba(0, 240, 118, 0.15)"

        st.markdown(f"""
        <div style="margin-bottom: 12px; background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #f1f5f9; font-size: 0.92rem;">{sig_name}</span>
                <div>
                    <span style="background: {badge_bg}; color: {bar_color}; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">{badge_text}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {bar_color}; font-size: 0.88rem;">{sig_score:.2f}</span>
                </div>
            </div>
            <div style="height: 6px; width: 100%; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; margin-bottom: 6px;">
                <div style="height: 100%; width: {int(sig_score * 100)}%; background: {bar_color}; border-radius: 3px;"></div>
            </div>
            <div style="color: #8a99ad; font-size: 0.80rem;">{sig_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Technical Details Expander
    with st.expander("🔬 Technical Telemetry & Decomposition"):
        st.markdown(f"""
        - **Frame Details:** `{sel_seg.get('technical_details', '')}`
        - **Raw Classification Flag:** `{sel_seg.get('reason', 'N/A')}`
        - **Full Technical Finding:** `{xai_data.get('technical_explanation', '')}`
        """)

    # Forensic Disclaimer
    st.markdown("""
    <div style="background: rgba(255, 170, 0, 0.08); border-left: 4px solid #ffaa00; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 0.85rem; color: #ffcc66;">
        ⚠️ <strong>Forensic Disclaimer:</strong> Highlighted temporal segments indicate model attribution and should be interpreted as supporting evidence, not proof of manipulation.
    </div>
    """, unsafe_allow_html=True)


def render_pillar3_audio_xai_section(xai_data: dict, audio_bytes: bytes = None):
    """
    Renders Explainable AI (XAI) section for Pillar 3 Audio Deepfake Forensics in Streamlit.
    Features:
      - Integrated Gradients Saliency Spectrogram
      - Highlighted Contributing Audio Segments
      - Audio Playback
      - Plain-English Human Explanation & Technical Telemetry
    """
    if not xai_data or not isinstance(xai_data, dict):
        return

    st.markdown("### 🎙️ Why this audio prediction? (Wav2Vec2 Integrated Gradients)")

    if not xai_data.get("xai_available", False):
        st.warning("⚠️ **Explanation unavailable for this sample.** The standard acoustic prediction remains valid.")
        return

    prediction = xai_data.get("prediction", "REAL")
    confidence = float(xai_data.get("confidence", 95.0))
    human_exp = xai_data.get("human_explanation", "")
    tech_exp = xai_data.get("technical_explanation", "")
    important_segments = xai_data.get("important_segments", [])

    st.info(f"💡 **Attribution Summary:** {human_exp}")

    # Audio Playback
    if audio_bytes:
        st.audio(audio_bytes)

    # Time-Frequency Saliency Spectrogram
    visualizations = xai_data.get("visualizations", [])
    vis = visualizations[0] if visualizations else {}
    saliency_url = vis.get("saliency_url") or xai_data.get("saliency_image")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saliency_file_path = os.path.join(base_dir, "Pillar 2", "video-authenticity-detector", "storage", saliency_url) if saliency_url else None

    if saliency_file_path and os.path.exists(saliency_file_path):
        st.markdown("##### 🔬 [Time-Frequency Saliency & Mel Spectrogram Heatmap]")
        sal_img = Image.open(saliency_file_path)
        st.image(sal_img, use_container_width=True, caption=f"Wav2Vec2 Integrated Gradients Saliency Map • Target: {prediction} ({confidence:.1f}%)")

    # Important Contributing Segments
    if important_segments:
        st.markdown("##### 📍 Highlighted Contributing Audio Segments")
        cols = st.columns(min(4, max(1, len(important_segments))))
        for idx, seg in enumerate(important_segments):
            with cols[idx % len(cols)]:
                is_fake = (seg.get("direction") == "supports_fake")
                border_color = "#ff3366" if is_fake else "#00f076"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid {border_color}; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #f1f5f9;">
                        #{seg.get('segment_id', idx+1)}: {seg.get('start_time', 0.0):.2f}s – {seg.get('end_time', 0.0):.2f}s
                    </div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: {border_color}; margin-top: 4px;">
                        {seg.get('attribution_percent', 50)}% ATTRIBUTION
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Accordions
    with st.expander("❓ What influenced this prediction?"):
        st.markdown(f"""
        - **Model:** Wav2Vec2 self-supervised speech transformer (`Hemgg/Deepfake-audio-detection`).
        - **Integrated Gradients:** Tracks gradient flow from raw audio waveform samples through self-attention layers to classification logits.
        - **Highlighted intervals** indicate specific time-frequency regions where acoustic artifacts (e.g. neural vocoder phase jitter, formant transitions) contributed to the **{prediction}** prediction.
        """)

    with st.expander("🔬 Technical Telemetry & Axioms"):
        st.markdown(f"""
        - **Technical Finding:** `{tech_exp}`
        - **Method:** Integrated Gradients (Sundararajan et al., 2017) with m=20 Riemann steps
        - **Baseline:** Silent zero-amplitude tensor (x'=0)
        """)

    # Forensic Disclaimer
    st.markdown("""
    <div style="background: rgba(255, 170, 0, 0.08); border-left: 4px solid #ffaa00; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 0.85rem; color: #ffcc66;">
        ⚠️ <strong>Forensic Disclaimer:</strong> Highlighted audio segments and spectrogram saliency reflect model attribution and should be interpreted as supporting evidence, not independent proof of manipulation.
    </div>
    """, unsafe_allow_html=True)


def render_pillar4_document_xai_section(xai_data: dict, p4_res: dict = None):
    """
    Renders Explainable AI (XAI) section for Pillar 4 Document & Benford Forensics in Streamlit.
    Features:
      - Benford Distribution Comparison Chart
      - Largest Deviations Callouts
      - MAE, Chi-Square & p-value Metric Tiles
      - Plain-English Human Explanation & Technical Telemetry
      - Sparse Sample Preservation Handling
    """
    if not xai_data or not isinstance(xai_data, dict):
        return

    st.markdown("### 📄 Why was this document flagged? (Statistical Explainability)")

    if not xai_data.get("xai_available", False) or xai_data.get("is_sparse", False):
        sparse_count = xai_data.get("digits_count", 0)
        st.warning(f"⚠️ **Statistical Explanation Limited (Sparse Document):** Document contains {sparse_count} numerical values (minimum 5 required, ≥15 recommended). Significance is not fabricated.")
        return

    prediction = xai_data.get("prediction", "AUTHENTIC DOCUMENT")
    confidence = float(xai_data.get("confidence", 85.0))
    mae = xai_data.get("mae")
    chi_sq = xai_data.get("chi_square")
    p_val = xai_data.get("p_value")
    total_n = xai_data.get("digits_count", 0)
    human_exp = xai_data.get("human_explanation", "")
    tech_exp = xai_data.get("technical_explanation", "")
    top_devs = xai_data.get("top_deviations", [])
    table_rows = xai_data.get("comparison_table", [])

    # Metric Badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("1. Numerical Values", f"N = {total_n}")
    with col2:
        st.metric("2. Mean Abs Error (MAE)", f"{mae:.4f}" if mae is not None else "—")
    with col3:
        st.metric("3. Chi-Square (χ²)", f"{chi_sq:.2f}" if chi_sq is not None else "—", delta="df = 8")
    with col4:
        p_str = f"{p_val:.4e}" if (p_val is not None and p_val < 0.001) else (f"{p_val:.4f}" if p_val is not None else "—")
        st.metric("4. p-value", p_str)

    # Explanation summary banner
    st.info(f"💡 **Statistical Forensic Summary:** {human_exp}")

    # Visual Chart
    chart_url = xai_data.get("chart_image")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chart_file_path = os.path.join(base_dir, "Pillar 2", "video-authenticity-detector", "storage", chart_url) if chart_url else None

    if chart_file_path and os.path.exists(chart_file_path):
        from PIL import Image
        st.markdown("##### 📊 [Benford First-Digit Distribution Chart]")
        chart_img = Image.open(chart_file_path)
        st.image(chart_img, use_container_width=True, caption=f"Observed First Digits vs. Theoretical Benford Law • {prediction} ({confidence:.1f}%)")

    # Top Deviations
    if top_devs:
        st.markdown("##### 📍 Largest Observed Deviations")
        dev_cols = st.columns(min(3, len(top_devs)))
        for i, dev in enumerate(top_devs[:3]):
            with dev_cols[i]:
                is_over = dev.get("diff_pct", 0) > 0
                badge_col = "#ff3366" if is_over else "#38bdf8"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid {badge_col}; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 800; color: #f1f5f9;">
                        Digit '{dev.get('digit')}'
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                        Obs: {dev.get('observed_pct')}% vs Exp: {dev.get('expected_pct')}%
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {badge_col}; margin-top: 4px;">
                        Δ = {dev.get('diff_pct'):+.1f}% ({'OVER' if is_over else 'UNDER'})
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Accordions
    with st.expander("❓ What does Benford's Law measure?"):
        st.markdown("""
        - **Benford's Law (First-Digit Law):** In naturally occurring numerical datasets (such as invoices, ledger entries, transactional receipts), numbers begin with '1' approximately 30.1% of the time, scaling logarithmically down to 4.6% for digit '9'.
        - **Forensic Application:** Synthetic generators, fabrication, or arbitrary digit manipulation tend to produce uniform or clustered digit distributions, creating high MAE and Chi-Square discrepancies.
        """)

    with st.expander("🔬 Technical Telemetry & Statistical Goodness-of-Fit"):
        st.markdown(f"""
        - **Technical Finding:** `{tech_exp}`
        - **Probability Mass Function (PMF):** `P(d) = log10(1 + 1/d)` for `d ∈ [1..9]`
        - **Degrees of Freedom:** `df = 8`
        - **Statistical Criterion:** Chi-Square goodness-of-fit with sample-size scaled thresholding
        """)

    # Non-Causal Forensic Notice
    st.markdown("""
    <div style="background: rgba(255, 170, 0, 0.08); border-left: 4px solid #ffaa00; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 0.85rem; color: #ffcc66;">
        ⚠️ <strong>Forensic Evidence Notice:</strong> This statistical pattern is consistent with the forensic signal detected by the system and should be considered alongside the other evidence.
    </div>
    """, unsafe_allow_html=True)


def render_pillar5_physics_xai_section(xai_data: dict):
    """
    Renders the Pillar 5 TreeSHAP physical forensics & steganalysis explainability section in Streamlit.
    """
    import os
    if not xai_data or not isinstance(xai_data, dict):
        return

    st.markdown("### 🔬 Why did the physical-forensics model make this prediction?")

    if not xai_data.get("xai_available", False):
        st.warning("⚠️ **TreeSHAP Unavailable:** Physical forensic feature attribution could not be extracted for this input.")
        return

    prediction = xai_data.get("prediction", "AUTHENTIC")
    confidence = float(xai_data.get("confidence", 85.0))
    human_exp = xai_data.get("human_explanation", "")
    tech_exp = xai_data.get("technical_explanation", "")
    model_name = xai_data.get("shap_model_used", "XGBoost Primary Tree Estimator")
    base_val = xai_data.get("base_value", 0.0)
    out_margin = xai_data.get("model_output_margin", 0.0)
    handcrafted = xai_data.get("handcrafted_forensic_features", [])

    # Badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("1. Model Explainer", model_name)
    with col2:
        st.metric("2. Base Value E[f(x)]", f"{base_val:.3f}")
    with col3:
        st.metric("3. Margin f(x)", f"{out_margin:+.3f}")

    st.info(f"💡 **Forensic Attribution:** {human_exp}")

    # Waterfall Plot
    plot_rel_path = xai_data.get("waterfall_plot_url")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_file_path = os.path.join(base_dir, "Pillar 2", "video-authenticity-detector", "storage", plot_rel_path) if plot_rel_path else None

    if plot_file_path and os.path.exists(plot_file_path):
        from PIL import Image
        st.markdown("##### 📊 [SHAP Waterfall Plot — Decision Flow]")
        plot_img = Image.open(plot_file_path)
        st.image(plot_img, use_container_width=True, caption=f"TreeSHAP Waterfall Plot ({model_name}) • {prediction} ({confidence:.1f}%)")

    # Handcrafted physics features table
    if handcrafted:
        st.markdown("##### 🔍 Handcrafted Forensic Features (Physics & SRM Steganalysis)")
        for feat in handcrafted:
            f_name = feat.get("feature", "")
            f_val = feat.get("value", 0.0)
            f_shap = feat.get("shap_value", 0.0)
            f_dir = feat.get("direction", "")
            f_desc = feat.get("human_description", "")
            is_fake = f_dir == "supports_fake"
            b_col = "#ff3366" if is_fake else "#00f076"

            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border-left: 4px solid {b_col}; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #f1f5f9;">{f_name}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {b_col};">SHAP: {f_shap:+.3f}</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{f_desc}</div>
                <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
                    Raw Value: {f_val:.4f} • Direction: <strong style="color: {b_col};">{'Pushes toward FAKE' if is_fake else 'Pushes toward REAL'}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("❓ How does TreeSHAP explain the physical model?"):
        st.markdown("""
        - **Positive Contribution (+):** Increases the log-odds margin $f(x)$ toward the Synthetic/Fake classification.
        - **Negative Contribution (−):** Decreases the log-odds margin $f(x)$ away from Synthetic (pushes toward Authentic/Real).
        - **Additive Exactness:** The model prediction decomposes exactly as $f(x) = E[f(x)] + \\sum \\phi_i$.
        """)

    with st.expander("🔬 Technical Telemetry & Feature Selection"):
        st.markdown(f"""
        - **Telemetry:** `{tech_exp}`
        - **Active Pipeline Features:** {xai_data.get('total_selected_features', 160)} selected from 1,304 total features (24 physics/SRM + 1,280 EfficientNet-B0 embeddings).
        """)





