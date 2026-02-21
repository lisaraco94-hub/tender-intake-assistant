"""
Tender Intake Assistant — Pre-Bid Screening
AI-powered analysis for TLA/IVD clinical laboratory tenders.
"""

import json
import os
import streamlit as st

from src.pipeline import read_pdf_pages, build_prebid_report, load_risk_factors
from src.report_docx import build_docx

st.set_page_config(page_title="Tender Intake Assistant", page_icon="🔬", layout="wide")

# ─── Brand style ──────────────────────────────────────────────────
try:
    with open("assets/brand_style.json", "r", encoding="utf-8") as f:
        brand = json.load(f)
except FileNotFoundError:
    brand = {"brand_name": "Inpeco", "primary_hex": "#003B6F", "accent_hex": "#00A0DC"}

st.title("🔬 Tender Intake Assistant")
st.caption("Pre-bid screening for TLA/IVD clinical laboratory tenders · Upload PDF → AI analysis → Word report")

# ─── API Key ──────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.warning("⚠️ Please provide your OpenAI API key in the sidebar to proceed.")
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**Brand:** {brand['brand_name']}")
    st.color_picker("Primary", brand["primary_hex"], disabled=True)
    st.color_picker("Accent", brand["accent_hex"], disabled=True)
    st.divider()

    st.markdown("### ⚙️ Analysis Settings")
    detail = st.select_slider("Detail level", options=["Low", "Medium", "High"], value="Medium")

    st.divider()
    st.markdown("### 📋 Risk Register")
    custom_rf = st.file_uploader("Upload custom risk_factors.json", type=["json"])

    if custom_rf:
        try:
            risk_factors = json.loads(custom_rf.read().decode("utf-8"))
            st.success("✅ Custom risk register loaded")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            risk_factors = load_risk_factors()
    else:
        try:
            risk_factors = load_risk_factors()
            st.info("Using default Inpeco risk register")
        except FileNotFoundError:
            st.error("assets/risk_factors.json not found")
            st.stop()

# ─── Main layout ──────────────────────────────────────────────────
col_input, col_output = st.columns([1, 2])

with col_input:
    st.markdown("### 📄 Input")
    pdf_file = st.file_uploader("Upload tender PDF", type=["pdf"])
    if pdf_file:
        st.success(f"✅ {pdf_file.name} ({pdf_file.size // 1024} KB)")
    run = st.button("🚀 Run Analysis", disabled=not pdf_file, use_container_width=True, type="primary")

with col_output:
    st.markdown("### 📊 Report")
    if not pdf_file:
        st.info("Upload a tender PDF to begin.")

# ─── Analysis ─────────────────────────────────────────────────────
if pdf_file and run:
    with st.spinner("Reading PDF..."):
        pages = read_pdf_pages(pdf_file.read())

    with st.spinner(f"Analyzing {len(pages)} pages with GPT-4o..."):
        try:
            report = build_prebid_report(pages, risk_factors=risk_factors, detail=detail)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    with col_output:
        rec = report.get("go_nogo", {}).get("recommendation", "")
        score = report.get("go_nogo", {}).get("score", 0)
        icon = {"GO": "🟢", "GO with Mitigation": "🟡", "NO-GO": "🔴"}.get(rec, "⚪")
        st.markdown(f"## {icon} {rec}  —  Complexity {score}/100")
        st.caption(report.get("go_nogo", {}).get("rationale", ""))

        st.markdown("### Executive Summary")
        for line in report.get("executive_summary", []):
            st.markdown(f"- {line}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tender Type", report.get("tender_type", "—").upper())
        c2.metric("Deadline", report.get("submission_deadline", "—"))
        c3.metric("Est. Value", report.get("estimated_value_eur", "—"))

    # Showstoppers
    showstoppers = report.get("showstoppers", [])
    if showstoppers:
        st.error(f"🚨 {len(showstoppers)} SHOWSTOPPER(S) identified")
        for ss in showstoppers:
            with st.expander(f"🚫 [{ss.get('id')}] {ss.get('description', '')}"):
                st.markdown(f"**Evidence:** {ss.get('evidence', '—')}")
                st.markdown(f"**Impact:** {ss.get('impact', '—')}")

    # Risk register
    risks = report.get("risks", [])
    if risks:
        st.markdown("### ⚠️ Risk Register")
        import pandas as pd
        df = pd.DataFrame(risks)
        cols = [c for c in ["id", "risk", "category", "probability", "impact", "score", "evidence", "mitigation"] if c in df.columns]
        st.dataframe(df[cols].sort_values("score", ascending=False), use_container_width=True, hide_index=True)

    # Requirements
    reqs = report.get("requirements", {})
    if reqs:
        st.markdown("### 📋 Requirements")
        tabs = st.tabs([k.replace("_", " ").title() for k in reqs.keys()])
        for tab, (key, items) in zip(tabs, reqs.items()):
            with tab:
                for item in (items or []):
                    st.markdown(f"- {item}")

    # Milestones
    deadlines = report.get("deadlines", [])
    if deadlines:
        st.markdown("### 📅 Key Milestones")
        for d in deadlines:
            st.markdown(f"- **{d.get('when', '?')}** — {d.get('milestone', '')}")

    # Open questions
    open_qs = report.get("open_questions", [])
    if open_qs:
        st.markdown("### ❓ Open Questions")
        for q in open_qs:
            st.markdown(f"- {q}")

    # Token cost
    meta = report.get("_meta", {})
    if meta:
        with st.expander("💰 API usage"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total tokens", f"{meta.get('total_tokens', 0):,}")
            c2.metric("Pages analyzed", meta.get("pages_analyzed", 0))
            c3.metric("Cost (USD)", f"${meta.get('estimated_cost_usd', 0):.4f}")

    # Download
    st.markdown("---")
    try:
        docx_bytes = build_docx(report, brand["primary_hex"], brand["accent_hex"])
        st.download_button(
            "📥 Download Word Report",
            data=docx_bytes,
            file_name=f"Tender_{report.get('tender_title','Report')[:40].replace(' ','_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            type="primary",
        )
    except Exception as e:
        st.warning(f"Could not generate Word report: {e}")
