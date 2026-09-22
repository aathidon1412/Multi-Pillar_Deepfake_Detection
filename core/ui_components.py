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
