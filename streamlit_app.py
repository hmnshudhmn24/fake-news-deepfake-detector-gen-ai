# streamlit_app.py
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="Fake News & Deepfake Detector", page_icon="🕵️", layout="wide")
st.title("🕵️ Fake News & Deepfake Detector (gen-ai)")

from src.article_checker import analyze_article
from src.video_checker import analyze_video

st.sidebar.header("Settings")
max_claims = st.sidebar.number_input("Max claims to verify", value=int(os.getenv("MAX_CLAIMS", 6)), min_value=1, max_value=20, step=1)

tab1, tab2 = st.tabs(["📰 Article / Claim Checker", "🎞️ Video Deepfake Detector"])

with tab1:
    st.header("Article / Claim Verification")
    input_type = st.radio("Input type", ["Paste article text", "Article URL"])
    text = ""
    url = ""
    if input_type == "Paste article text":
        text = st.text_area("Paste article text here", height=300)
    else:
        url = st.text_input("Article URL")

    if st.button("🔎 Analyze article"):
        with st.spinner("Extracting claims and verifying..."):
            result = analyze_article(text=text, url=url, max_claims=max_claims)
        st.subheader("Summary")
        st.write(f"Article-level verdict: **{result.get('article_verdict','unknown')}**")
        st.markdown("### Claims & verdicts")
        for c in result.get("claims", []):
            st.markdown(f"- **Claim**: {c['claim_text']}")
            st.markdown(f"  - Verdict: **{c['verdict']}** (confidence: {c.get('confidence', 0):.2f})")
            if c.get("evidence"):
                st.markdown("  - Evidence:")
                for e in c["evidence"][:3]:
                    st.write(f"    - {e.get('title','')} — {e.get('url','')}")
        st.markdown("### LLM explanation")
        st.write(result.get("llm_summary", ""))

with tab2:
    st.header("Video Deepfake Detector")
    uploaded = st.file_uploader("Upload video file (mp4/mov)", type=["mp4", "mov", "mkv"])
    sample_rate = st.number_input("Frames to analyze per second", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    run_btn = st.button("🔬 Analyze video")

    if run_btn:
        if not uploaded:
            st.error("Please upload a video file.")
        else:
            tmp_dir = os.path.join("temp_uploads", )
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            with st.spinner("Extracting frames and running detection... This may take a while."):
                vres = analyze_video(tmp_path, fps=sample_rate)
            st.subheader("Video report")
            st.write(f"Video-level fake probability: **{vres.get('fake_prob', 0.0):.2f}**")
            st.markdown("Per-frame scores (top suspicious frames):")
            for item in vres.get("top_frames", [])[:8]:
                try:
                    st.image(item["frame_path"], caption=f"score={item['score']:.3f}", width=240)
                except Exception:
                    st.write(item)
            if vres.get("notes"):
                st.info(vres["notes"])
