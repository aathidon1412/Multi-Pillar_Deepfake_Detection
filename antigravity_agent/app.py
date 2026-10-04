"""
Streamlit Web Application for antigravity_agent
Interactive multimodal deepfake detection dashboard with real-time timeline graph and explainability report.
"""

import os
import tempfile
import streamlit as st
from antigravity_agent.core.orchestrator import OrchestratorAgent

st.set_page_config(
    page_title="Antigravity Deepfake Agent",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Antigravity Agent: Autonomous Deepfake Detection")
st.markdown("Unified multimodal forensic analysis across spatial artifacts, temporal optical flow, and acoustic integrity.")

@st.cache_resource
def get_agent():
    return OrchestratorAgent()

agent = get_agent()

uploaded_file = st.file_uploader("Upload an MP4 Video for Forensic Evaluation", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_video_path = tmp.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Video Preview")
        st.video(tmp_video_path)

    if st.button("🚀 Run Forensic Agent Inspection", type="primary"):
        with st.spinner("Executing multi-pillar forensic agent pipeline..."):
            temp_out_dir = tempfile.mkdtemp()
            result = agent.run(tmp_video_path, output_dir=temp_out_dir)

        consensus = result["consensus"]
        report = result["report"]
        meta = result["metadata"]

        verdict = consensus["verdict"]
        prob = consensus["deepfake_probability"]

        st.divider()

        # Top Metric Banner
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        if verdict == "DEEPFAKE":
            m_col1.error(f"Verdict: **{verdict}**")
        elif verdict == "REAL":
            m_col1.success(f"Verdict: **{verdict}**")
        else:
            m_col1.warning(f"Verdict: **{verdict}**")

        m_col2.metric("Deepfake Probability", f"{prob * 100:.1f}%")
        m_col3.metric("Confidence Level", consensus["confidence_level"])
        m_col4.metric("Analyzed Frames", meta["total_frames"])

        st.markdown(f"**Forensic Summary:** *{consensus['description']}*")

        # Two columns for Details & Timeline
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.subheader("📈 Anomaly Timeline (Spatial vs. Motion Flow)")
            if os.path.exists(report["timeline_chart"]):
                st.image(report["timeline_chart"], caption="Multimodal Anomaly Timeline", use_container_width=True)

        with c_right:
            st.subheader("📋 Itemized Forensic Evidence")
            for item in report["itemized_findings"]:
                st.write(f"- {item}")

            st.subheader("📊 Individual Module Scores")
            contribs = consensus["pillar_contributions"]
            st.progress(float(contribs["spatial"]), text=f"Spatial Anomaly Score: {contribs['spatial'] * 100:.1f}%")
            st.progress(float(contribs["temporal"]), text=f"Temporal Divergence: {contribs['temporal'] * 100:.1f}%")
            if contribs["audio"] is not None:
                st.progress(float(contribs["audio"]), text=f"Audio Anomaly Score: {contribs['audio'] * 100:.1f}%")
