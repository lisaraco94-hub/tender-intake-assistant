"""
Tender Intake Assistant — Pre-Bid Screening
AI-powered analysis for TLA/IVD clinical laboratory tenders.
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.extractors import extract_from_file, SUPPORTED_EXTENSIONS
from src.pipeline import build_prebid_report, load_risk_factors
from src.report_docx import build_docx

# ─── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Inpeco · Tender Intake",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Brand colours ────────────────────────────────────────────────
PRIMARY  = "#00AEEF"   # Inpeco cyan
ORANGE   = "#F7941D"   # Inpeco orange
NAVY     = "#003865"   # Inpeco navy
LIGHT_BG = "#F4F7FA"
WHITE    = "#FFFFFF"

# ─── Library persistence ─────────────────────────────────────────
LIBRARY_PATH = "assets/tender_library.json"


def load_library() -> list:
    if os.path.exists(LIBRARY_PATH):
        try:
            with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_to_library(entry: dict):
    lib = load_library()
    lib.insert(0, entry)
    with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=2)


RISK_FACTORS_PATH = "assets/risk_factors.json"


def save_risk_factors(rf: dict):
    with open(RISK_FACTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(rf, f, ensure_ascii=False, indent=2)


def _load_knowledge_context(max_chars_per_file: int = 12_000, max_total: int = 36_000) -> str:
    """Load text from past bid response documents (won and lost) stored in the knowledge base."""
    parts = []
    total = 0
    for folder, label in [
        ("won",  "WON TENDER — Inpeco's Response"),
        ("lost", "LOST TENDER — Inpeco's Response"),
    ]:
        folder_path = f"assets/knowledge/{folder}"
        if not os.path.exists(folder_path):
            continue
        for fn in sorted(os.listdir(folder_path)):
            if total >= max_total:
                break
            fp = os.path.join(folder_path, fn)
            try:
                with open(fp, "rb") as f:
                    file_bytes = f.read()
                pages = extract_from_file(file_bytes, fn)
                text = "\n".join(pages)[:max_chars_per_file]
                parts.append(f"=== {label}: {fn} ===\n{text}")
                total += len(text)
            except Exception:
                pass
    return "\n\n".join(parts)


def _ai_format_risk(concept: str, entry_type: str, rf: dict) -> dict:
    """Convert a plain-language risk description into a structured JSON entry using GPT-4o."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    if entry_type == "showstopper":
        existing = rf.get("risk_register", {}).get("showstoppers", [])
        prefix = "SS"
    else:
        existing = rf.get("risk_register", {}).get("risk_factors", [])
        prefix = "HR"

    max_n = 0
    for e in existing:
        try:
            max_n = max(max_n, int(e.get("id", "0").split("-")[-1]))
        except Exception:
            pass
    next_id = f"{prefix}-{max_n + 1:02d}"

    if entry_type == "showstopper":
        schema = (
            f'{{"id": "{next_id}", "name": "short name (max 8 words)", '
            '"description": "clear explanation of why this is a showstopper", '
            '"signals": ["keyword or phrase 1", "keyword or phrase 2", "keyword or phrase 3"]}}'
        )
        context = "This is a SHOWSTOPPER — a reason to immediately decline to bid."
    else:
        schema = (
            f'{{"id": "{next_id}", "name": "short name (max 8 words)", '
            '"description": "clear explanation of the risk", '
            '"signals": ["signal phrase 1", "signal phrase 2"], '
            '"category": "e.g. Technical / Commercial / Legal / Operational", '
            '"probability": 3, "impact": 3}}'
        )
        context = "This is a HIGH RISK factor — something that significantly complicates the bid."

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a tender risk analyst for Inpeco (Total Laboratory Automation supplier). "
                    f"{context} "
                    f"Convert the user's plain-text description into a structured entry. "
                    f"Return ONLY valid JSON exactly matching this schema: {schema}"
                ),
            },
            {"role": "user", "content": concept},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


# ─── Session state ────────────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state.view = "home"
if "report" not in st.session_state:
    st.session_state.report = None
if "run_done" not in st.session_state:
    st.session_state.run_done = False
if "detail" not in st.session_state:
    st.session_state.detail = "Medium"

# ─── Global CSS + Montserrat ──────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after {{
    font-family: 'Montserrat', sans-serif !important;
}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.stMainBlockContainer,
.block-container {{
    background: {PRIMARY} !important;
}}

section[data-testid="stSidebar"] {{
    background: {PRIMARY} !important;
}}

[data-testid="collapsedControl"] {{
    display: none;
}}

/* ── Navbar ── */
.top-nav {{
    background: transparent;
    padding: 2.2rem 2.8rem 1.4rem;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    margin-bottom: 1.5rem;
}}
.nav-logo-block {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.35rem;
}}
.nav-tagline {{
    font-size: 0.78rem;
    font-weight: 300;
    color: rgba(255,255,255,0.88);
    letter-spacing: 0.05em;
    padding-left: 0.1rem;
}}

/* ── Hero ── */
.hero {{
    text-align: center;
    padding: 1rem 2rem 2rem;
}}
.hero-title {{
    font-size: 2.4rem;
    font-weight: 800;
    color: {WHITE};
    line-height: 1.2;
    margin-bottom: 0.55rem;
}}
.hero-title span {{
    color: {WHITE};
    opacity: 0.82;
}}
.hero-sub {{
    font-size: 0.96rem;
    color: rgba(255,255,255,0.8);
    font-weight: 400;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.75;
}}

/* ── Feature cards ── */
.feat-card {{
    background: {WHITE};
    border-radius: 18px;
    padding: 2.4rem 2rem 1.6rem;
    text-align: center;
    box-shadow: 0 2px 14px rgba(0,0,0,0.065);
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.6rem;
    transition: transform 0.22s, box-shadow 0.22s, border-color 0.22s;
}}
.feat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, {PRIMARY}, {ORANGE});
}}
.feat-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 16px 40px rgba(0,56,101,0.13);
    border-color: {PRIMARY};
}}
.feat-icon {{
    font-size: 2.8rem;
    margin-bottom: 1rem;
    display: block;
}}
.feat-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {NAVY};
    margin-bottom: 0.5rem;
}}
.feat-desc {{
    font-size: 0.82rem;
    color: #6a8aaa;
    line-height: 1.65;
}}
.feat-badge {{
    display: inline-block;
    background: {PRIMARY}22;
    color: {PRIMARY};
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.feat-badge-orange {{
    background: {ORANGE}22;
    color: {ORANGE};
}}

/* ── Section headings ── */
.section-heading {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {NAVY};
    border-left: 4px solid {PRIMARY};
    padding-left: 0.75rem;
    margin: 1.6rem 0 1rem;
    letter-spacing: 0.01em;
}}
.section-heading-orange {{
    border-left-color: {ORANGE};
}}

/* ── Verdict banners ── */
.verdict-go {{
    background: linear-gradient(135deg, #e6f9f0, #d4f5e5);
    border-left: 5px solid #27ae60;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
}}
.verdict-go-mit {{
    background: linear-gradient(135deg, #fff8e6, #fef0c7);
    border-left: 5px solid {ORANGE};
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
}}
.verdict-nogo {{
    background: linear-gradient(135deg, #fff0f0, #fde8e8);
    border-left: 5px solid #e74c3c;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
}}
.verdict-label {{
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}}
.verdict-score {{
    font-size: 0.85rem;
    opacity: 0.72;
    margin-top: 0.3rem;
}}
.verdict-rationale {{
    font-size: 0.85rem;
    margin-top: 0.6rem;
    opacity: 0.88;
    font-style: italic;
    line-height: 1.55;
}}

/* ── Showstopper card ── */
.ss-card {{
    background: #fff5f5;
    border: 1.5px solid #e74c3c;
    border-radius: 10px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.6rem;
}}
.ss-id {{
    font-size: 0.7rem;
    font-weight: 700;
    color: #c0392b;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}
.ss-desc {{
    font-size: 0.92rem;
    font-weight: 600;
    color: #1a2d42;
    margin: 0.25rem 0;
}}
.ss-evidence {{
    font-size: 0.78rem;
    color: #666;
    line-height: 1.5;
}}

/* ── Library rows ── */
.lib-row {{
    background: {WHITE};
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.55rem;
    border-left: 4px solid {PRIMARY};
    box-shadow: 0 1px 8px rgba(0,0,0,0.055);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}}
.lib-row-nogo {{ border-left-color: #e74c3c; }}
.lib-row-mit  {{ border-left-color: {ORANGE}; }}
.lib-title    {{ font-size: 0.9rem; font-weight: 700; color: {NAVY}; }}
.lib-meta     {{ font-size: 0.74rem; color: #7a96b0; margin-top: 0.2rem; }}
.lib-summary  {{ font-size: 0.78rem; color: #4a6a8a; margin-top: 0.3rem; font-style: italic; }}
.lib-badge {{
    font-size: 0.69rem;
    font-weight: 700;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
}}
.badge-go   {{ background: #d4f5e5; color: #1a7a48; }}
.badge-mit  {{ background: #fef0c7; color: #a06000; }}
.badge-nogo {{ background: #fde8e8; color: #c0392b; }}

/* ── Info / warn boxes ── */
.info-box {{
    background: linear-gradient(135deg, #e8f4fb, #f0f7fb);
    border-left: 3px solid {PRIMARY};
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #2a4a62;
    margin: 0.5rem 0;
    line-height: 1.65;
}}
.warn-box {{
    background: #fff8e6;
    border-left: 3px solid {ORANGE};
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #7a5200;
    margin: 0.5rem 0;
}}
.trunc-warn {{
    background: #fff8e6;
    border-left: 3px solid {ORANGE};
    border-radius: 4px;
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
    color: #7a5200;
    margin-bottom: 0.6rem;
}}

/* ── File tags ── */
.file-tag {{
    display: inline-block;
    background: #e8f4fb;
    border: 1px solid {PRIMARY};
    border-radius: 20px;
    padding: 0.2rem 0.72rem;
    font-size: 0.72rem;
    color: {NAVY};
    margin: 0.15rem 0.2rem;
    font-weight: 500;
}}

/* ── Metric containers ── */
[data-testid="metric-container"] {{
    background: {WHITE} !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    border: 1px solid #dde8f0 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
}}

/* ── Buttons ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #0090cc 100%) !important;
    color: {WHITE} !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Montserrat', sans-serif !important;
}}
.stButton > button[kind="secondary"] {{
    border: 2px solid {PRIMARY} !important;
    color: {PRIMARY} !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-family: 'Montserrat', sans-serif !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {NAVY} !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: rgba(255,255,255,0.85) !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-color: rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.12) !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {{
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    font-family: 'Montserrat', sans-serif !important;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 3.5rem 2rem;
}}
.empty-icon {{ font-size: 3rem; }}
.empty-msg  {{ font-size: 0.9rem; color: #8aa5c0; margin-top: 0.8rem; line-height: 1.7; }}
</style>
""", unsafe_allow_html=True)


# ─── Shared navbar ────────────────────────────────────────────────
# Inpeco logo — white SVG on one line (no Markdown code-block issue)
# Large filled circle + small dot upper-left + "inpeco" bold wordmark
_INPECO_LOGO_SVG = '<svg height="52" viewBox="0 0 255 52" xmlns="http://www.w3.org/2000/svg"><circle cx="26" cy="33" r="22" fill="white"/><circle cx="8" cy="8" r="7.5" fill="white"/><text x="57" y="44" font-family="Montserrat,Arial,sans-serif" font-weight="900" font-size="34" fill="white">inpeco</text></svg>'

def _nav(view_name: str):
    st.markdown(
        '<div class="top-nav"><div class="nav-logo-block">'
        + _INPECO_LOGO_SVG
        + '<div class="nav-tagline">Creating a healthier tomorrow, today</div>'
        + '</div></div>',
        unsafe_allow_html=True,
    )


# ─── HOME ─────────────────────────────────────────────────────────
def view_home():
    _nav("home")

    st.markdown("""
    <div class="hero">
      <div class="hero-title">Tender Intake <span>Assistant</span></div>
      <div class="hero-sub">
        AI-powered pre-bid screening for TLA/IVD clinical laboratory tenders.<br>
        Analyse risks, extract requirements, and make faster Go/No-Go decisions.
      </div>
    </div>
    """, unsafe_allow_html=True)

    lib_count = len(load_library())
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="feat-card">
          <span class="feat-icon">📋</span>
          <div class="feat-title">Analyse Tender</div>
          <div class="feat-desc">
            Upload tender documents, run AI analysis, and receive a full pre-bid report
            with Go/No-Go recommendation, risks, requirements, and milestones.
          </div>
          <div class="feat-badge">AI · GPT-4o</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_analyze", use_container_width=True, type="primary"):
            st.session_state.view = "analyze"
            st.session_state.run_done = False
            st.rerun()

    with c2:
        st.markdown(f"""
        <div class="feat-card">
          <span class="feat-icon">📚</span>
          <div class="feat-title">Tender Library</div>
          <div class="feat-desc">
            Browse every analysed tender — date, client, country, verdict, and a
            one-line summary. Track your history and export as CSV.
          </div>
          <div class="feat-badge feat-badge-orange">{lib_count} tender{"s" if lib_count != 1 else ""}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_library", use_container_width=True):
            st.session_state.view = "library"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="feat-card">
          <span class="feat-icon">🧠</span>
          <div class="feat-title">Knowledge Base</div>
          <div class="feat-desc">
            Upload custom showstoppers, risk registers, and past won/lost tenders
            to continuously improve screening accuracy.
          </div>
          <div class="feat-badge feat-badge-orange">Configurable</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_knowledge", use_container_width=True):
            st.session_state.view = "knowledge"
            st.rerun()

    # Footer bar
    st.markdown(f"""
    <div style="text-align:center;margin-top:3rem;padding:1.5rem;border-top:1px solid #dde8f0;">
      <span style="font-size:0.72rem;color:#a0b8cc;letter-spacing:0.04em;">
        INPECO · Tender Intake Assistant · Powered by GPT-4o
        &nbsp;·&nbsp;
        <span style="color:{ORANGE};">●</span>&nbsp;Active
      </span>
    </div>
    """, unsafe_allow_html=True)


# ─── ANALYSE ──────────────────────────────────────────────────────
def view_analyze():
    _nav("analyze")

    if st.button("← Back to Home", key="back_analyze"):
        st.session_state.view = "home"
        st.session_state.run_done = False
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.success("API key active ✓")

        st.divider()
        st.markdown("**Analysis depth**")
        detail = st.select_slider(
            "depth",
            options=["Low", "Medium", "High"],
            value=st.session_state.detail,
            label_visibility="collapsed",
        )
        st.session_state.detail = detail
        st.caption({
            "Low":    "~1–2 min · Showstoppers + top 3 risks",
            "Medium": "~2–4 min · Full risk + requirements",
            "High":   "~4–8 min · Exhaustive analysis",
        }[detail])

    # Upload
    st.markdown('<div class="section-heading">Upload Tender Documents</div>', unsafe_allow_html=True)
    accepted = sorted(SUPPORTED_EXTENSIONS)
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=accepted,
        accept_multiple_files=True,
        help=f"Accepted: {', '.join('.' + e for e in accepted)}",
    )

    with st.expander("Custom risk register (optional)"):
        custom_rf_file = st.file_uploader(
            "Upload risk_factors.json", type=["json"], key="rf_uploader"
        )

    # Load risk factors
    if custom_rf_file:
        try:
            risk_factors = json.loads(custom_rf_file.read().decode("utf-8"))
            st.success("Custom risk register loaded.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            risk_factors = load_risk_factors()
    else:
        try:
            risk_factors = load_risk_factors()
        except FileNotFoundError:
            st.error("assets/risk_factors.json not found.")
            st.stop()

    if uploaded_files:
        st.markdown(
            " ".join(f'<span class="file-tag">{f.name}</span>' for f in uploaded_files),
            unsafe_allow_html=True,
        )
        total_kb = sum(f.size for f in uploaded_files) // 1024
        st.caption(f"{len(uploaded_files)} file(s) · {total_kb:,} KB")

        can_run = bool(os.environ.get("OPENAI_API_KEY", ""))
        if not can_run:
            st.markdown(
                '<div class="warn-box">⚠️ Enter your OpenAI API key in the sidebar to run analysis.</div>',
                unsafe_allow_html=True,
            )

        if st.button("🔍 Run Analysis", disabled=not can_run, type="primary"):
            all_pages: list[str] = []
            with st.spinner("Reading files…"):
                for uf in uploaded_files:
                    pages = extract_from_file(uf.read(), uf.name)
                    all_pages.append(f"=== FILE: {uf.name} ===")
                    all_pages.extend(pages)

            knowledge_ctx = _load_knowledge_context()
            n_kb = len([
                fn
                for folder in ("assets/knowledge/won", "assets/knowledge/lost")
                if os.path.exists(folder)
                for fn in os.listdir(folder)
            ])
            spinner_msg = (
                f"Analysing {len(uploaded_files)} file(s) with GPT-4o [{detail}]"
                + (f" · {n_kb} past bid doc(s) loaded" if n_kb else "")
                + "…"
            )
            with st.spinner(spinner_msg):
                try:
                    report = build_prebid_report(
                        all_pages,
                        risk_factors=risk_factors,
                        detail=detail,
                        knowledge_context=knowledge_ctx,
                    )
                    st.session_state.report = report
                    st.session_state.run_done = True

                    summary_line = (report.get("executive_summary") or [""])[0][:140]
                    save_to_library({
                        "date":    datetime.now().strftime("%Y-%m-%d"),
                        "title":   report.get("tender_title") or "—",
                        "client":  report.get("contracting_authority") or "—",
                        "country": report.get("country") or "—",
                        "verdict": report.get("go_nogo", {}).get("recommendation", "—"),
                        "score":   report.get("go_nogo", {}).get("score", 0),
                        "summary": summary_line,
                        "files":   [f.name for f in uploaded_files],
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    if st.session_state.run_done and st.session_state.report:
        _render_report(st.session_state.report)


# ─── LIBRARY ──────────────────────────────────────────────────────
def view_library():
    _nav("library")

    if st.button("← Back to Home", key="back_library"):
        st.session_state.view = "home"
        st.rerun()

    st.markdown('<div class="section-heading">Tender Library</div>', unsafe_allow_html=True)

    lib = load_library()
    if not lib:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📭</div>
          <div class="empty-msg">No tenders analysed yet.<br>
          Run your first analysis to start building the library.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analysed", len(lib))
    c2.metric("GO", sum(1 for e in lib if e.get("verdict") == "GO"))
    c3.metric("GO w/ Mitigation", sum(1 for e in lib if "Mitigation" in (e.get("verdict") or "")))
    c4.metric("NO-GO", sum(1 for e in lib if e.get("verdict") == "NO-GO"))

    st.markdown('<div class="section-heading" style="margin-top:1.8rem;">All Tenders</div>', unsafe_allow_html=True)

    search = st.text_input(
        "search",
        placeholder="🔍  Filter by title, client, country…",
        label_visibility="collapsed",
    )
    filtered = lib
    if search:
        q = search.lower()
        filtered = [
            e for e in lib
            if q in (
                (e.get("title") or "") +
                (e.get("client") or "") +
                (e.get("country") or "") +
                (e.get("summary") or "")
            ).lower()
        ]

    st.caption(f"{len(filtered)} result(s)")

    for entry in filtered:
        v = entry.get("verdict", "—")
        row_cls   = "lib-row" + (" lib-row-nogo" if v == "NO-GO" else " lib-row-mit" if "Mitigation" in v else "")
        badge_cls = "badge-nogo" if v == "NO-GO" else ("badge-mit" if "Mitigation" in v else "badge-go")
        st.markdown(f"""
        <div class="{row_cls}">
          <div style="flex:1;min-width:0;">
            <div class="lib-title">{entry.get("title","—")}</div>
            <div class="lib-meta">
              📅 {entry.get("date","—")} &nbsp;·&nbsp;
              🏢 {entry.get("client","—")} &nbsp;·&nbsp;
              🌍 {entry.get("country","—")} &nbsp;·&nbsp;
              Score: {entry.get("score","—")}/100
            </div>
            <div class="lib-summary">{entry.get("summary","")}</div>
          </div>
          <div><span class="lib-badge {badge_cls}">{v}</span></div>
        </div>
        """, unsafe_allow_html=True)

    if filtered:
        st.divider()
        df_e = pd.DataFrame(filtered)
        cols = [c for c in ["date", "title", "client", "country", "verdict", "score", "summary"] if c in df_e.columns]
        st.download_button(
            "⬇️ Export as CSV",
            data=df_e[cols].to_csv(index=False),
            file_name="tender_library.csv",
            mime="text/csv",
        )


# ─── KNOWLEDGE BASE ───────────────────────────────────────────────
def view_knowledge():
    _nav("knowledge")

    if st.button("← Back to Home", key="back_knowledge"):
        st.session_state.view = "home"
        st.rerun()

    st.markdown('<div class="section-heading">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      Enrich the assistant's analysis by uploading your own risk factors, showstoppers,
      and past tender examples. The more context you provide, the more accurate
      the screening results will be.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📁  Risk Factors & Showstoppers",
        "🏆  Won Tenders",
        "❌  Lost Tenders",
    ])

    with tab1:
        can_add = bool(os.environ.get("OPENAI_API_KEY", ""))

        # ── Add a new risk/showstopper with AI ──────────────────────
        st.markdown('<div class="section-heading">Add Risk or Showstopper with AI</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          Describe a risk or showstopper in plain language — the AI will structure it
          automatically and add it to the active register. No JSON knowledge required.
        </div>
        """, unsafe_allow_html=True)

        if not can_add:
            st.markdown(
                '<div class="warn-box">⚠️ Enter your OpenAI API key in the Analyse Tender sidebar first.</div>',
                unsafe_allow_html=True,
            )

        entry_type = st.radio(
            "Type to add",
            ["Showstopper (reason to decline immediately)", "Risk factor (something that complicates the bid)"],
            horizontal=True,
            key="kb_entry_type",
        )
        is_ss = entry_type.startswith("Showstopper")

        concept = st.text_area(
            "Describe the risk in plain language",
            placeholder=(
                "e.g. Sometimes the tender requires a connection to a specific middleware brand "
                "that we have never integrated before and the timeline is too short to develop it."
                if not is_ss else
                "e.g. The tender specifies that the system must be the same brand currently installed "
                "in their lab, which is a competitor."
            ),
            height=100,
            key="kb_concept",
        )

        if st.button("✨ Add with AI", disabled=(not can_add or not concept.strip()), type="primary", key="kb_add_ai"):
            with st.spinner("AI is structuring the entry…"):
                try:
                    rf = load_risk_factors()
                    new_entry = _ai_format_risk(concept.strip(), "showstopper" if is_ss else "risk_factor", rf)
                    rr = rf.setdefault("risk_register", {})
                    if is_ss:
                        rr.setdefault("showstoppers", []).append(new_entry)
                    else:
                        rr.setdefault("risk_factors", []).append(new_entry)
                    save_risk_factors(rf)
                    st.success(f"Added **{new_entry.get('id')} — {new_entry.get('name')}**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        # ── Current register ─────────────────────────────────────────
        st.markdown('<div class="section-heading" style="margin-top:2rem;">Active Risk Register</div>', unsafe_allow_html=True)
        try:
            rf = load_risk_factors()
            ss_list = rf.get("risk_register", {}).get("showstoppers") or rf.get("showstoppers", [])
            rf_list = rf.get("risk_register", {}).get("risk_factors") or rf.get("risk_factors", [])

            st.caption(f"{len(ss_list or [])} showstoppers · {len(rf_list or [])} risk factors")

            with st.expander(f"🚨 Showstoppers ({len(ss_list or [])})"):
                for i, ss in enumerate(ss_list or []):
                    c_txt, c_del = st.columns([10, 1])
                    c_txt.markdown(f"**{ss.get('id','')}** — {ss.get('name','')}  \n*{ss.get('description','')}*")
                    if c_del.button("🗑", key=f"del_ss_{i}", help="Remove"):
                        rf["risk_register"]["showstoppers"].pop(i)
                        save_risk_factors(rf)
                        st.rerun()

            with st.expander(f"⚠️ Risk factors ({len(rf_list or [])})"):
                for i, r in enumerate(rf_list or []):
                    c_txt, c_del = st.columns([10, 1])
                    c_txt.markdown(f"**{r.get('id','')}** — {r.get('name','')}  \n*{r.get('description','')}*")
                    if c_del.button("🗑", key=f"del_rf_{i}", help="Remove"):
                        rf["risk_register"]["risk_factors"].pop(i)
                        save_risk_factors(rf)
                        st.rerun()
        except Exception as e:
            st.warning(f"Could not load risk register: {e}")

        # ── Advanced: replace entire register via JSON upload ────────
        with st.expander("Advanced: replace entire register via JSON upload"):
            rf_up = st.file_uploader("Upload risk_factors.json", type=["json"], key="kb_rf")
            if rf_up:
                try:
                    data = json.loads(rf_up.read().decode("utf-8"))
                    save_risk_factors(data)
                    n_ss = len(data.get("showstoppers", []) or data.get("risk_register", {}).get("showstoppers", []))
                    n_rf = len(data.get("risk_factors", []) or data.get("risk_register", {}).get("risk_factors", []))
                    st.success(f"Risk register replaced — {n_ss} showstoppers, {n_rf} risk factors.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.markdown('<div class="section-heading">Won Tenders — Inpeco\'s Bid Responses</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <b>Upload Inpeco's written responses / technical offers for tenders you WON.</b><br><br>
          The AI reads these to learn <em>what Inpeco successfully committed to</em> — proven capabilities,
          typical delivery timelines, commercial terms that worked, and the language used when confident.
          These documents are automatically included in every future analysis.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
          📌 <b>Upload Inpeco's response document</b>, not the original tender.
          The response is what reveals your real capabilities and commitments.
        </div>
        """, unsafe_allow_html=True)
        won_ups = st.file_uploader(
            "Upload Inpeco response documents (PDF, DOCX, TXT…)",
            type=sorted(SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            key="kb_won",
        )
        if won_ups:
            os.makedirs("assets/knowledge/won", exist_ok=True)
            for f in won_ups:
                with open(f"assets/knowledge/won/{f.name}", "wb") as out:
                    out.write(f.getvalue())
            st.success(f"{len(won_ups)} file(s) added to Won knowledge base.")
        won_dir = "assets/knowledge/won"
        if os.path.exists(won_dir) and os.listdir(won_dir):
            files = sorted(os.listdir(won_dir))
            st.markdown(f"**Stored ({len(files)} response document(s)):**")
            for fn in files:
                c_fn, c_del = st.columns([10, 1])
                c_fn.markdown(f"- `{fn}`")
                if c_del.button("🗑", key=f"del_won_{fn}", help="Remove"):
                    os.remove(os.path.join(won_dir, fn))
                    st.rerun()
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <div class="empty-msg">No won-tender responses uploaded yet.</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-heading">Lost Tenders — Inpeco\'s Bid Responses</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <b>Upload Inpeco's written responses / technical offers for tenders you LOST.</b><br><br>
          The AI looks for the soft or hedged language that often signals real limitations —
          phrases like <em>"subject to site survey"</em>, <em>"to be confirmed at kick-off"</em>,
          <em>"in principle compatible"</em>. These patterns reveal where Inpeco struggles even
          when the response appears compliant on the surface.
          These documents are automatically included in every future analysis.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
          📌 <b>Upload Inpeco's response document</b>, not the original tender.
          The response — with its diplomatic wording — is what exposes the real gaps.
        </div>
        """, unsafe_allow_html=True)
        lost_ups = st.file_uploader(
            "Upload Inpeco response documents (PDF, DOCX, TXT…)",
            type=sorted(SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            key="kb_lost",
        )
        if lost_ups:
            os.makedirs("assets/knowledge/lost", exist_ok=True)
            for f in lost_ups:
                with open(f"assets/knowledge/lost/{f.name}", "wb") as out:
                    out.write(f.getvalue())
            st.success(f"{len(lost_ups)} file(s) added to Lost knowledge base.")
        lost_dir = "assets/knowledge/lost"
        if os.path.exists(lost_dir) and os.listdir(lost_dir):
            files = sorted(os.listdir(lost_dir))
            st.markdown(f"**Stored ({len(files)} response document(s)):**")
            for fn in files:
                c_fn, c_del = st.columns([10, 1])
                c_fn.markdown(f"- `{fn}`")
                if c_del.button("🗑", key=f"del_lost_{fn}", help="Remove"):
                    os.remove(os.path.join(lost_dir, fn))
                    st.rerun()
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <div class="empty-msg">No lost-tender responses uploaded yet.</div>
            </div>
            """, unsafe_allow_html=True)


# ─── REPORT RENDERER ──────────────────────────────────────────────
def _render_report(report: dict):
    detail = st.session_state.get("detail", "Medium")
    meta   = report.get("_meta", {})

    if meta.get("truncated"):
        st.markdown(
            f'<div class="trunc-warn">⚠️ Document truncated to fit the <b>{detail}</b> limit. '
            f'Switch to <b>High</b> for complete analysis.</div>',
            unsafe_allow_html=True,
        )

    # Verdict banner
    go_nogo   = report.get("go_nogo", {})
    rec       = go_nogo.get("recommendation", "—")
    score     = go_nogo.get("score", 0)
    rationale = go_nogo.get("rationale", "")
    icons   = {"GO": "✅", "GO with Mitigation": "⚠️", "NO-GO": "🚫"}
    classes = {"GO": "verdict-go", "GO with Mitigation": "verdict-go-mit", "NO-GO": "verdict-nogo"}
    st.markdown(f"""
    <div class="{classes.get(rec, 'verdict-go')}">
      <div class="verdict-label">{icons.get(rec, "⚪")} {rec}</div>
      <div class="verdict-score">Complexity score: {score} / 100</div>
      <div class="verdict-rationale">{rationale}</div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tender type",  (report.get("tender_type") or "—").upper())
    c2.metric("Deadline",      report.get("submission_deadline") or "—")
    c3.metric("Est. value",    report.get("estimated_value_eur") or "—")
    c4.metric("Authority",    (report.get("contracting_authority") or "—")[:30])

    # Executive summary
    st.markdown('<div class="section-heading">Executive Summary</div>', unsafe_allow_html=True)
    for line in report.get("executive_summary", []):
        st.markdown(f"- {line}")

    # Showstoppers
    showstoppers = report.get("showstoppers", [])
    if showstoppers:
        st.markdown(
            f'<div class="section-heading section-heading-orange">🚨 Showstoppers ({len(showstoppers)})</div>',
            unsafe_allow_html=True,
        )
        for ss in showstoppers:
            st.markdown(f"""
<div class="ss-card">
  <div class="ss-id">{ss.get("id", "SS")}</div>
  <div class="ss-desc">{ss.get("description", "")}</div>
  <div class="ss-evidence">Evidence: {ss.get("evidence", "—")} · Impact: {ss.get("impact", "—")}</div>
</div>""", unsafe_allow_html=True)

    # Tabbed detail
    risks        = report.get("risks", [])
    reqs         = report.get("requirements", {})
    deadlines    = report.get("deadlines", [])
    deliverables = report.get("deliverables", [])
    open_qs      = report.get("open_questions", [])

    t_risk, t_req, t_mile, t_del, t_q = st.tabs([
        f"Risks ({len(risks)})",
        "Requirements",
        f"Milestones ({len(deadlines)})",
        f"Deliverables ({len(deliverables)})",
        f"Open Questions ({len(open_qs)})",
    ])

    with t_risk:
        if risks:
            df = pd.DataFrame(risks)
            ordered = [c for c in ["id","risk","category","probability","impact","score","evidence","mitigation"] if c in df.columns]
            st.dataframe(df[ordered].sort_values("score", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No risks identified.")

    with t_req:
        if reqs:
            sub = st.tabs([k.replace("_", " ").title() for k in reqs])
            for tab, (key, items) in zip(sub, reqs.items()):
                with tab:
                    for item in (items or []):
                        st.markdown(f"- {item}")
                    if not items:
                        st.caption("Nothing detected.")
        else:
            st.info("No requirements extracted.")

    with t_mile:
        for d in (deadlines or []):
            st.markdown(f"- **{d.get('when','?')}** — {d.get('milestone','')}  \n  *{d.get('evidence','')}*")
        if not deadlines:
            st.info("No milestones detected.")

    with t_del:
        for item in (deliverables or []):
            st.markdown(f"- {item}")
        if not deliverables:
            st.info("No deliverables listed.")

    with t_q:
        for q in (open_qs or []):
            st.markdown(f"- {q}")
        if not open_qs:
            st.info("No open questions flagged.")

    # Download + API usage
    st.divider()
    col_dl, col_meta = st.columns([2, 1])

    with col_dl:
        try:
            docx_bytes = build_docx(report, PRIMARY, ORANGE)
            safe = (report.get("tender_title") or "Report")[:40].replace(" ", "_")
            st.download_button(
                "⬇️ Download Word Report",
                data=docx_bytes,
                file_name=f"Tender_Intake_{safe}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                type="primary",
            )
        except Exception as e:
            st.warning(f"Could not generate Word report: {e}")

    with col_meta:
        if meta:
            with st.expander("API usage"):
                st.caption(f"Model: {meta.get('model','gpt-4o')}")
                st.caption(f"Tokens: {meta.get('total_tokens',0):,}")
                st.caption(f"Cost: ${meta.get('estimated_cost_usd',0):.4f}")
                st.caption(f"Sections: {meta.get('pages_analyzed',0)}")
                if meta.get("truncated"):
                    st.caption(f"⚠️ Truncated at {meta.get('chars_analyzed',0):,} chars")


# ─── Router ───────────────────────────────────────────────────────
_view = st.session_state.view
if   _view == "home":      view_home()
elif _view == "analyze":   view_analyze()
elif _view == "library":   view_library()
elif _view == "knowledge": view_knowledge()
else:
    st.session_state.view = "home"
    st.rerun()
