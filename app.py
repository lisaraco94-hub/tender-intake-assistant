"""
Tender Intake Assistant — Pre-Bid Screening
AI-powered analysis for TLA/IVD clinical laboratory tenders.
"""

import json
import os

import pandas as pd
import streamlit as st

from src.extractors import extract_from_file, SUPPORTED_EXTENSIONS
from src.pipeline import build_prebid_report, load_risk_factors
from src.report_docx import build_docx

# ─── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Inpeco · Tender Intake",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Brand style ──────────────────────────────────────────────────
try:
    with open("assets/brand_style.json", "r", encoding="utf-8") as f:
        brand = json.load(f)
except FileNotFoundError:
    brand = {"brand_name": "Inpeco", "primary_hex": "#33bce5", "accent_hex": "#f3b08f"}

PRIMARY = brand["primary_hex"]
ACCENT = brand["accent_hex"]
NAVY = "#0d2b45"

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Header ── */
.inpeco-header {{
    background: linear-gradient(135deg, {NAVY} 0%, #1c4a6e 100%);
    padding: 1.4rem 2rem 1.2rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}}
.inpeco-wordmark {{
    font-size: 2rem;
    font-weight: 800;
    color: {PRIMARY};
    letter-spacing: 0.12em;
    line-height: 1;
}}
.inpeco-tagline {{
    font-size: 0.78rem;
    color: #8db4cc;
    margin-top: 0.3rem;
    letter-spacing: 0.03em;
}}
.header-divider {{
    width: 2px;
    height: 2.8rem;
    background: {PRIMARY};
    opacity: 0.4;
    border-radius: 2px;
}}
.header-app-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #e8f4fb;
}}
.header-app-sub {{
    font-size: 0.75rem;
    color: #8db4cc;
    margin-top: 0.2rem;
}}

/* ── Verdict banner ── */
.verdict-go {{
    background: linear-gradient(90deg, #e6f9f0, #d4f5e5);
    border-left: 5px solid #27ae60;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}}
.verdict-go-mit {{
    background: linear-gradient(90deg, #fff8e6, #fef0c7);
    border-left: 5px solid #f0a500;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}}
.verdict-nogo {{
    background: linear-gradient(90deg, #fff0f0, #fde8e8);
    border-left: 5px solid #e74c3c;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}}
.verdict-label {{
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}}
.verdict-score {{
    font-size: 0.85rem;
    opacity: 0.75;
    margin-top: 0.2rem;
}}
.verdict-rationale {{
    font-size: 0.82rem;
    margin-top: 0.5rem;
    opacity: 0.85;
    font-style: italic;
}}

/* ── Showstopper card ── */
.ss-card {{
    background: #fff5f5;
    border: 1.5px solid #e74c3c;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}}
.ss-id {{ font-size: 0.72rem; font-weight: 700; color: #c0392b; letter-spacing: 0.05em; }}
.ss-desc {{ font-size: 0.9rem; font-weight: 600; color: #1a2d42; margin: 0.2rem 0; }}
.ss-evidence {{ font-size: 0.78rem; color: #555; }}

/* ── Section headings ── */
.section-heading {{
    font-size: 1rem;
    font-weight: 700;
    color: {NAVY};
    border-bottom: 2px solid {PRIMARY};
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.8rem 0;
    letter-spacing: 0.01em;
}}

/* ── File tag ── */
.file-tag {{
    display: inline-block;
    background: #e8f4fb;
    border: 1px solid {PRIMARY};
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: {NAVY};
    margin: 0.15rem 0.2rem;
}}

/* ── Detail info box ── */
.detail-info {{
    background: #f0f7fb;
    border-left: 3px solid {PRIMARY};
    border-radius: 4px;
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: #2a4a62;
    margin-top: 0.4rem;
}}

/* ── Sidebar tweaks ── */
[data-testid="stSidebar"] {{
    background: #f5f9fd;
}}
[data-testid="stSidebar"] hr {{
    border-color: #cce3f0;
}}

/* ── Welcome card ── */
.welcome-card {{
    background: linear-gradient(135deg, #f0f7fb, #ffffff);
    border: 1px solid #cce3f0;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: {NAVY};
}}
.welcome-icon {{ font-size: 3rem; }}
.welcome-title {{ font-size: 1.2rem; font-weight: 700; margin: 0.8rem 0 0.4rem 0; }}
.welcome-sub {{ font-size: 0.85rem; color: #4a7a99; }}

/* ── Truncation warning ── */
.trunc-warn {{
    background: #fff8e6;
    border-left: 3px solid {ACCENT};
    border-radius: 4px;
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
    color: #7a5200;
    margin-bottom: 0.5rem;
}}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="inpeco-header">
    <div>
        <div class="inpeco-wordmark">INPECO</div>
        <div class="inpeco-tagline">Total Laboratory Automation</div>
    </div>
    <div class="header-divider"></div>
    <div>
        <div class="header-app-title">Tender Intake Assistant</div>
        <div class="header-app-sub">Pre-bid screening · AI-powered · TLA/IVD</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    # API Key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    # Detail level
    st.markdown("**Analysis detail level**")
    detail = st.select_slider(
        "detail",
        options=["Low", "Medium", "High"],
        value="Medium",
        label_visibility="collapsed",
    )
    detail_meta = {
        "Low":    ("~1–2 min", "Showstoppers + top 3 risks only. Ideal for bulk screening of many files."),
        "Medium": ("~2–4 min", "Full risk assessment and requirements extraction."),
        "High":   ("~4–8 min", "Exhaustive: all constraints, open questions, full evidence."),
    }
    dur, desc = detail_meta[detail]
    st.markdown(f'<div class="detail-info"><b>{dur}</b> — {desc}</div>', unsafe_allow_html=True)

    st.divider()

    # File upload
    st.markdown("**Tender documents**")
    accepted_types = sorted(SUPPORTED_EXTENSIONS)
    uploaded_files = st.file_uploader(
        "files",
        type=accepted_types,
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=f"Accepted: {', '.join('.' + e for e in accepted_types)}",
    )

    if uploaded_files:
        total_kb = sum(f.size for f in uploaded_files) // 1024
        st.markdown(
            " ".join(
                f'<span class="file-tag">{f.name}</span>'
                for f in uploaded_files
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"{len(uploaded_files)} file(s) · {total_kb:,} KB total")

    st.divider()

    # Risk register
    st.markdown("**Risk register**")
    custom_rf = st.file_uploader(
        "Custom risk_factors.json (optional)",
        type=["json"],
        label_visibility="visible",
    )
    if custom_rf:
        try:
            risk_factors = json.loads(custom_rf.read().decode("utf-8"))
            st.success("Custom register loaded")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            risk_factors = load_risk_factors()
    else:
        try:
            risk_factors = load_risk_factors()
            st.caption("Using default Inpeco register")
        except FileNotFoundError:
            st.error("assets/risk_factors.json not found")
            st.stop()

    st.divider()

    # Run button
    can_run = bool(uploaded_files) and bool(os.environ.get("OPENAI_API_KEY", ""))
    run = st.button(
        "Run Analysis",
        disabled=not can_run,
        use_container_width=True,
        type="primary",
    )
    if not os.environ.get("OPENAI_API_KEY", ""):
        st.caption("Enter your OpenAI API key above to proceed.")
    elif not uploaded_files:
        st.caption("Upload at least one tender document.")

# ─── Main area — idle state ───────────────────────────────────────
if not uploaded_files or not run:
    st.markdown(f"""
    <div class="welcome-card">
        <div class="welcome-icon">📋</div>
        <div class="welcome-title">Upload tender documents to begin</div>
        <div class="welcome-sub">
            Supported formats: {", ".join("." + e for e in sorted(SUPPORTED_EXTENSIONS))}<br>
            You can upload multiple files from the same tender package.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Extract text from all uploaded files ─────────────────────────
all_pages: list[str] = []
file_summary: list[str] = []

with st.spinner("Reading files..."):
    for uf in uploaded_files:
        raw = uf.read()
        pages = extract_from_file(raw, uf.name)
        # Insert a file boundary marker so GPT-4o knows which file content is from
        all_pages.append(f"=== FILE: {uf.name} ===")
        all_pages.extend(pages)
        file_summary.append(f"{uf.name} ({len(pages)} pages/chunks)")

total_pages = len(all_pages)

# ─── Run AI analysis ──────────────────────────────────────────────
with st.spinner(f"Analyzing {len(uploaded_files)} file(s) · {total_pages} sections · GPT-4o [{detail}]..."):
    try:
        report = build_prebid_report(all_pages, risk_factors=risk_factors, detail=detail)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ─── Results ──────────────────────────────────────────────────────
meta = report.get("_meta", {})

# Truncation warning
if meta.get("truncated"):
    st.markdown(
        f'<div class="trunc-warn">⚠️ Document was truncated to fit the <b>{detail}</b> detail limit. '
        f'Switch to <b>High</b> for complete analysis.</div>',
        unsafe_allow_html=True,
    )

# ── GO / NO-GO verdict ────────────────────────────────────────────
go_nogo = report.get("go_nogo", {})
rec = go_nogo.get("recommendation", "—")
score = go_nogo.get("score", 0)
rationale = go_nogo.get("rationale", "")

verdict_icons = {"GO": "✅", "GO with Mitigation": "⚠️", "NO-GO": "🚫"}
verdict_classes = {"GO": "verdict-go", "GO with Mitigation": "verdict-go-mit", "NO-GO": "verdict-nogo"}
icon = verdict_icons.get(rec, "⚪")
cls = verdict_classes.get(rec, "verdict-go")

st.markdown(f"""
<div class="{cls}">
    <div class="verdict-label">{icon} {rec}</div>
    <div class="verdict-score">Complexity score: {score} / 100</div>
    <div class="verdict-rationale">{rationale}</div>
</div>
""", unsafe_allow_html=True)

# ── Key metadata ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tender type", (report.get("tender_type") or "—").upper())
c2.metric("Submission deadline", report.get("submission_deadline") or "—")
c3.metric("Est. value", report.get("estimated_value_eur") or "—")
c4.metric("Authority", (report.get("contracting_authority") or "—")[:30])

# ── Executive summary ─────────────────────────────────────────────
st.markdown('<div class="section-heading">Executive Summary</div>', unsafe_allow_html=True)
for line in report.get("executive_summary", []):
    st.markdown(f"- {line}")

# ── Showstoppers ──────────────────────────────────────────────────
showstoppers = report.get("showstoppers", [])
if showstoppers:
    st.markdown(
        f'<div class="section-heading">🚨 Showstoppers ({len(showstoppers)})</div>',
        unsafe_allow_html=True,
    )
    for ss in showstoppers:
        st.markdown(f"""
<div class="ss-card">
    <div class="ss-id">{ss.get("id", "SS")}</div>
    <div class="ss-desc">{ss.get("description", "")}</div>
    <div class="ss-evidence">Evidence: {ss.get("evidence", "—")} · Impact: {ss.get("impact", "—")}</div>
</div>
""", unsafe_allow_html=True)

# ── Tabbed results ────────────────────────────────────────────────
risks = report.get("risks", [])
reqs = report.get("requirements", {})
deadlines = report.get("deadlines", [])
deliverables = report.get("deliverables", [])
open_qs = report.get("open_questions", [])

tab_risks, tab_reqs, tab_milestones, tab_deliverables, tab_questions = st.tabs([
    f"Risks ({len(risks)})",
    "Requirements",
    f"Milestones ({len(deadlines)})",
    f"Deliverables ({len(deliverables)})",
    f"Open Questions ({len(open_qs)})",
])

with tab_risks:
    if risks:
        df = pd.DataFrame(risks)
        ordered_cols = [c for c in ["id", "risk", "category", "probability", "impact", "score", "evidence", "mitigation"] if c in df.columns]
        st.dataframe(
            df[ordered_cols].sort_values("score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No risks identified.")

with tab_reqs:
    if reqs:
        req_tabs = st.tabs([k.replace("_", " ").title() for k in reqs.keys()])
        for rtab, (key, items) in zip(req_tabs, reqs.items()):
            with rtab:
                if items:
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.caption("Nothing detected in this category.")
    else:
        st.info("No requirements extracted.")

with tab_milestones:
    if deadlines:
        for d in deadlines:
            st.markdown(f"- **{d.get('when', '?')}** — {d.get('milestone', '')}  \n  *{d.get('evidence', '')}*")
    else:
        st.info("No explicit milestones detected.")

with tab_deliverables:
    if deliverables:
        for item in deliverables:
            st.markdown(f"- {item}")
    else:
        st.info("No deliverables listed.")

with tab_questions:
    if open_qs:
        for q in open_qs:
            st.markdown(f"- {q}")
    else:
        st.info("No open questions flagged.")

# ── Download + API usage ──────────────────────────────────────────
st.divider()
col_dl, col_meta = st.columns([2, 1])

with col_dl:
    try:
        docx_bytes = build_docx(report, PRIMARY, ACCENT)
        safe_title = (report.get("tender_title") or "Report")[:40].replace(" ", "_")
        st.download_button(
            "Download Word Report",
            data=docx_bytes,
            file_name=f"Tender_Intake_{safe_title}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            type="primary",
        )
    except Exception as e:
        st.warning(f"Could not generate Word report: {e}")

with col_meta:
    if meta:
        with st.expander("API usage"):
            st.caption(f"Model: {meta.get('model', 'gpt-4o')}")
            st.caption(f"Tokens: {meta.get('total_tokens', 0):,}")
            st.caption(f"Cost: ${meta.get('estimated_cost_usd', 0):.4f}")
            st.caption(f"Files: {len(uploaded_files)} · Sections: {meta.get('pages_analyzed', 0)}")
            if meta.get("truncated"):
                st.caption(f"⚠️ Truncated at {meta.get('chars_analyzed', 0):,} chars")
