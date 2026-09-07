import os
import sys
import tempfile
import gradio as gr
from detect import classify_audio

def predict_audio(audio_file, analysis_mode):
    """
    Takes an uploaded or recorded audio file path and mode ("spoken" or "music"):
    - Real vs Fake label dictionary with probabilities for Gradio Label component
    - Markdown formatted analysis details including music warning banners
    """
    if audio_file is None:
        return None, "⚠️ Please upload or record an audio file to analyze."

    try:
        if isinstance(audio_file, tuple):
            sr, data = audio_file
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            sf.write(temp_path, data, sr)
            file_path = temp_path
        elif isinstance(audio_file, str):
            file_path = audio_file
        else:
            file_path = getattr(audio_file, "name", str(audio_file))

        # Map UI choice to internal mode key
        internal_mode = "music" if "Song" in analysis_mode or "Music" in analysis_mode else "spoken"
        results = classify_audio(file_path, mode=internal_mode)

        # Gradio Label format: dict mapping class name to confidence (0.0 to 1.0)
        label_output = {
            "REAL (Human Voice)": round(results["real_confidence"] / 100.0, 4),
            "FAKE (AI Synthesized)": round(results["fake_confidence"] / 100.0, 4)
        }

        # Warning banner if music is detected while in Spoken Voice mode
        music_alert_html = ""
        if results["is_music"] and internal_mode == "spoken":
            music_alert_html = """
<div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.5rem;">⚠️</span>
        <h4 style="color: #92400E; margin: 0; font-size: 1.05rem; font-weight: 700;">
            Music / Heavy Instrumentation Detected
        </h4>
    </div>
    <p style="color: #78350F; margin: 6px 0 0 0; font-size: 0.95rem; line-height: 1.4;">
        Studio tracks with background beats, polyphonic instruments, and autotune cause false positives.
        The model was trained purely on conversational speech. Switch to <b>"🎵 Song / Music Track (Experimental / Demixed)"</b> mode for harmonic separation.
    </p>
</div>
"""

        # Disclaimer banner in Music mode
        if results.get("disclaimer"):
            music_alert_html += f"""
<div style="background-color: #EFF6FF; border-left: 5px solid #3B82F6; border-radius: 8px; padding: 12px 16px; margin: 10px 0;">
    <p style="color: #1E40AF; margin: 0; font-size: 0.95rem; font-weight: 500;">
        ℹ️ <b>Music Track Demixing Applied:</b> Harmonic vocal was isolated using HPSS to remove background percussion. Predictions reflect processed singing vocals.
    </p>
</div>
"""

        verdict_color = "#10B981" if results["prediction"] == "REAL" else "#EF4444"
        verdict_badge = f"""
<div style="background-color: {verdict_color}20; border: 2px solid {verdict_color}; border-radius: 12px; padding: 16px; margin-top: 10px; text-align: center;">
    <h2 style="color: {verdict_color}; margin: 0; font-size: 1.8rem; font-weight: 700;">
        VERDICT: {results['prediction']}
    </h2>
    <p style="margin: 6px 0 0 0; font-size: 1.1rem; color: #374151; font-weight: 500;">
        Top Confidence: <b>{results['confidence']:.2f}%</b>
    </p>
</div>
"""
        demix_badge = "✅ Applied (Harmonic Vocal Separated via HPSS)" if results.get("is_demixed") else "None (Raw Pass-Through)"
        music_flag = "🎵 Music / Beats Detected" if results["is_music"] else "🗣️ Pure Speech Voice"

        details_md = f"""
### 📊 Acoustic Forensics Breakdown
- **File Analysed**: `{os.path.basename(file_path)}`
- **Analysis Mode**: `{analysis_mode}`
- **Audio Classification**: `{music_flag}`
- **Audio Duration**: `{results['duration']:.2f} seconds`
- **Sample Rate**: `{results['samplerate']} Hz`
- **Harmonic Demixing**: `{demix_badge}`
- **Chunks Evaluated**: `{results.get('chunks_evaluated', 1)}`

| Classification Class | Probability / Confidence |
| :--- | :--- |
| **REAL (Human Voice)** | **{results['real_confidence']:.2f}%** |
| **FAKE (AI Voice / TTS / Cloned)** | **{results['fake_confidence']:.2f}%** |
"""
        return label_output, music_alert_html + verdict_badge + details_md

    except Exception as e:
        return None, f"❌ **Error processing audio:** {str(e)}"

CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.header-box {
    text-align: center;
    padding: 24px 10px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.header-box h1 {
    color: #38bdf8 !important;
    font-size: 2.2rem !important;
    margin-bottom: 8px !important;
    font-weight: 800;
}
.header-box p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}
"""

with gr.Blocks(title="Voice Deepfake Detector (Pillar 3)", css=CUSTOM_CSS) as demo:
    with gr.Column():
        gr.HTML("""
        <div class="header-box">
            <h1>🎙️ Voice Deepfake Detector</h1>
            <p>Pillar 3 • Acoustic & Speech Synthetic Voice Classification Engine</p>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Powered by Hugging Face <code>Hemgg/Deepfake-audio-detection</code> + Librosa HPSS Forensics</p>
        </div>
        """)

        mode_radio = gr.Radio(
            choices=[
                "🗣️ Spoken Voice / Phone Call (Standard)",
                "🎵 Song / Music Track (Experimental / Demixed)"
            ],
            value="🗣️ Spoken Voice / Phone Call (Standard)",
            label="🎯 Select Audio Type / Pipeline Mode:"
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="📤 Upload Audio or Record Speech (.wav, .mp3)"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")

                dummy_example_path = os.path.join("test_audio", "sample_test.wav")
                examples = [[dummy_example_path, "🗣️ Spoken Voice / Phone Call (Standard)"]] if os.path.exists(dummy_example_path) else None
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[audio_input, mode_radio],
                        label="🎵 Pre-loaded Test Audio"
                    )

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="Detection Probabilities (Confidence)",
                    num_top_classes=2
                )
                details_output = gr.HTML(
                    value="<p style='color: #64748b; text-align: center; padding: 20px;'>Upload an audio file and click <b>Analyze Audio</b> to inspect results.</p>"
                )

        analyze_btn.click(
            fn=predict_audio,
            inputs=[audio_input, mode_radio],
            outputs=[label_output, details_output]
        )
        audio_input.change(
            fn=predict_audio,
            inputs=[audio_input, mode_radio],
            outputs=[label_output, details_output]
        )
        mode_radio.change(
            fn=predict_audio,
            inputs=[audio_input, mode_radio],
            outputs=[label_output, details_output]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
