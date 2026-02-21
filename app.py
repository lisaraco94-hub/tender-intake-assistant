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
    """Load text from past bid response documents stored in the knowledge base."""
    parts = []
    total = 0
    for folder, label in [
        ("responses", "PAST BID RESPONSE — Inpeco"),
        # backward-compat with old won/lost folders
        ("won",  "PAST BID RESPONSE (won) — Inpeco"),
        ("lost", "PAST BID RESPONSE (lost) — Inpeco"),
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


# ─── Query-param card navigation ─────────────────────────────────
_qp = st.query_params.get("nav", "")
if _qp in ("analyze", "library", "knowledge"):
    st.query_params.clear()
    st.session_state.view = _qp
    st.rerun()

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
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

*, *::before, *::after {{
    font-family: 'Montserrat', sans-serif !important;
}}

/* ── Ripristina il font icone Streamlit (specificità > * vince su !important) ── */
[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] *,
[data-baseweb="tab-list"] button > span,
[data-baseweb="tab-list"] button > p,
[data-baseweb="tab-list"] button[aria-label] *,
[data-baseweb="accordion"] [role="button"] > div > span,
[data-baseweb="accordion"] [role="button"] > div > p,
summary > div > span:first-child,
button[aria-label*="arrow"] span,
button[aria-label*="arrow"] p,
span.material-symbols-rounded {{
    font-family: 'Material Symbols Rounded', 'Material Icons', serif !important;
    font-feature-settings: 'liga' 1;
    -webkit-font-feature-settings: 'liga' 1;
}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.stMainBlockContainer,
.block-container {{
    background: {PRIMARY} !important;
}}

[data-testid="stMainBlockContainer"],
.block-container {{
    padding-top: 0 !important;
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
    padding: 0.9rem 2.8rem 0.5rem;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    margin-bottom: 0.4rem;
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
    padding: 0.2rem 2rem 0.9rem;
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
[data-testid="stHorizontalBlock"] {{
    align-items: stretch !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    display: flex !important;
    flex-direction: column !important;
}}
a.feat-link {{
    text-decoration: none;
    display: flex;
    flex-direction: column;
    flex: 1;
    margin-bottom: 0.6rem;
}}
.feat-card {{
    background: {WHITE};
    border-radius: 18px;
    padding: 2.4rem 2rem 1.6rem;
    text-align: center;
    box-shadow: 0 2px 14px rgba(0,0,0,0.065);
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    flex: 1;
    transition: transform 0.22s, box-shadow 0.22s, border-color 0.22s;
}}
a.feat-link:hover .feat-card {{
    transform: translateY(-5px);
    box-shadow: 0 16px 40px rgba(0,56,101,0.13);
    border-color: {PRIMARY};
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

/* ── Section headings (on cyan background) ── */
.section-heading {{
    font-size: 1rem;
    font-weight: 700;
    color: white;
    border-left: 3px solid rgba(255,255,255,0.35);
    padding-left: 0.75rem;
    margin: 1.8rem 0 0.8rem;
    letter-spacing: 0.02em;
}}
.section-heading-orange {{
    border-left-color: {ORANGE};
}}

/* ── Verdict banners ── */
.verdict-go, .verdict-go-mit, .verdict-nogo {{
    background: rgba(255,255,255,0.13);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
    border-left-width: 5px;
    border-left-style: solid;
}}
.verdict-go     {{ border-left-color: #2ecc71; }}
.verdict-go-mit {{ border-left-color: {ORANGE}; }}
.verdict-nogo   {{ border-left-color: #e74c3c; }}
.verdict-label {{
    font-size: 1.55rem;
    font-weight: 800;
    color: white;
    letter-spacing: 0.05em;
}}
.verdict-score {{
    font-size: 0.85rem;
    color: rgba(255,255,255,0.72);
    margin-top: 0.3rem;
}}
.verdict-rationale {{
    font-size: 0.85rem;
    color: rgba(255,255,255,0.88);
    margin-top: 0.6rem;
    font-style: italic;
    line-height: 1.55;
}}

/* ── Showstopper card ── */
.ss-card {{
    background: rgba(255,255,255,0.1);
    border-left: 3px solid {ORANGE};
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.6rem;
}}
.ss-id {{
    font-size: 0.7rem;
    font-weight: 700;
    color: {ORANGE};
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}
.ss-desc {{
    font-size: 0.92rem;
    font-weight: 600;
    color: white;
    margin: 0.25rem 0;
}}
.ss-evidence {{
    font-size: 0.78rem;
    color: rgba(255,255,255,0.68);
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

/* ── Inline notes (plain text, no boxes) ── */
.info-box {{
    font-size: 0.83rem;
    color: rgba(255,255,255,0.72);
    margin: 0.2rem 0 1rem;
    line-height: 1.75;
}}
.warn-box {{
    font-size: 0.83rem;
    color: rgba(255,255,255,0.88);
    margin: 0.4rem 0 0.8rem;
    line-height: 1.6;
}}
.trunc-warn {{
    font-size: 0.79rem;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0.5rem;
}}

/* ── File tags ── */
.file-tag {{
    display: inline-block;
    background: rgba(255,255,255,0.88);
    border-radius: 20px;
    padding: 0.2rem 0.72rem;
    font-size: 0.72rem;
    color: {NAVY};
    margin: 0.15rem 0.2rem;
    font-weight: 500;
}}

/* ── Widget labels and captions on cyan ── */
.stCaption p {{
    color: rgba(255,255,255,0.6) !important;
}}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectSlider"] label,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stRadio"] label,
[data-testid="stRadio"] p,
[data-testid="stSelectbox"] label {{
    color: rgba(255,255,255,0.82) !important;
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

/* ── Tabs su sfondo azzurro ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: none !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    font-family: 'Montserrat', sans-serif !important;
    color: rgba(255,255,255,0.82) !important;
    border-radius: 7px !important;
    border: none !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    background: {WHITE} !important;
    color: {NAVY} !important;
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
# Large circle (r=25) fully inside viewBox; small dot upper-left (top y=3 safe)
_INPECO_LOGO_SVG = '<svg height="68" viewBox="0 0 290 68" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="39" r="25" fill="white"/><circle cx="10" cy="11" r="8" fill="white"/><text x="66" y="57" font-family="Montserrat,Arial,sans-serif" font-weight="900" font-size="41" fill="white">inpeco</text></svg>'

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
        <a class="feat-link" href="?nav=analyze">
          <div class="feat-card">
            <span class="feat-icon">📋</span>
            <div class="feat-title">Analyse Tender</div>
            <div class="feat-desc">
              Upload a tender document, run AI analysis and receive a full
              Go/No-Go report with risk register, requirements and milestones.
            </div>
            <div class="feat-badge">AI · GPT-4o</div>
          </div>
        </a>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <a class="feat-link" href="?nav=library">
          <div class="feat-card">
            <span class="feat-icon">📚</span>
            <div class="feat-title">Tender Library</div>
            <div class="feat-desc">
              Browse all analysed tenders — date, client, country, verdict
              and summary. Searchable and exportable as CSV.
            </div>
            <div class="feat-badge feat-badge-orange">{lib_count} tender{"s" if lib_count != 1 else ""}</div>
          </div>
        </a>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <a class="feat-link" href="?nav=knowledge">
          <div class="feat-card">
            <span class="feat-icon">🧠</span>
            <div class="feat-title">Knowledge Base</div>
            <div class="feat-desc">
              Manage evaluation rules and past bid responses
              to improve analysis accuracy over time.
            </div>
            <div class="feat-badge feat-badge-orange">Configurable</div>
          </div>
        </a>
        """, unsafe_allow_html=True)

    # Footer bar
    st.markdown(f"""
    <div style="text-align:center;margin-top:3rem;padding:1.5rem;border-top:1px solid rgba(255,255,255,0.15);">
      <span style="font-size:0.72rem;color:rgba(255,255,255,0.45);letter-spacing:0.04em;">
        INPECO · Tender Intake Assistant · Powered by GPT-4o
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

    # ── API key + depth — visible in main area ──────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "")
    col_key, col_depth = st.columns([3, 1], gap="large")
    with col_key:
        if not api_key:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-…",
                key="api_key_main",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.rerun()
        else:
            st.markdown(
                '<p style="color:rgba(255,255,255,0.75);font-size:0.82rem;margin:0.6rem 0;">🔑 API key active</p>',
                unsafe_allow_html=True,
            )
    with col_depth:
        detail = st.select_slider(
            "Analysis depth",
            options=["Low", "Medium", "High"],
            value=st.session_state.detail,
        )
        st.session_state.detail = detail
        st.caption({
            "Low":    "~1–2 min · showstoppers only",
            "Medium": "~2–4 min · risks + requirements",
            "High":   "~4–8 min · full analysis",
        }[detail])

    # Sidebar (still available for advanced users)
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if not os.environ.get("OPENAI_API_KEY"):
            sk = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="api_key_sidebar")
            if sk:
                os.environ["OPENAI_API_KEY"] = sk
        else:
            st.success("API key active ✓")

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
                '<p style="color:rgba(255,255,255,0.75);font-size:0.82rem;">Enter your OpenAI API key above to run the analysis.</p>',
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

    tab1, tab2 = st.tabs([
        "📁  Risk Factors & Showstoppers",
        "📄  Past Bid Responses",
    ])

    with tab1:
        can_add = bool(os.environ.get("OPENAI_API_KEY", ""))

        st.markdown('<div class="section-heading">Add entry with AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="info-box">Describe a new risk or showstopper in plain language. '
            'The AI will structure it and add it to the active register.</p>',
            unsafe_allow_html=True,
        )

        if not can_add:
            st.markdown(
                '<p class="warn-box">API key required — enter it in the Analyse Tender page first.</p>',
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

    with tab2:
        st.markdown('<div class="section-heading">Past Bid Responses</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="info-box">Upload documents containing Inpeco\'s written responses to past tenders — '
            'the AI reads them to understand real capabilities and identify patterns in how limitations '
            'are expressed. These files are automatically included in every new analysis.</p>',
            unsafe_allow_html=True,
        )

        resp_ups = st.file_uploader(
            "Upload response documents (PDF, DOCX, TXT…)",
            type=sorted(SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            key="kb_responses",
        )
        resp_dir = "assets/knowledge/responses"
        if resp_ups:
            os.makedirs(resp_dir, exist_ok=True)
            for f in resp_ups:
                with open(os.path.join(resp_dir, f.name), "wb") as out:
                    out.write(f.getvalue())
            st.success(f"{len(resp_ups)} file(s) added to the knowledge base.")

        if os.path.exists(resp_dir) and os.listdir(resp_dir):
            files = sorted(os.listdir(resp_dir))
            st.caption(f"{len(files)} document(s) stored")
            for fn in files:
                c_fn, c_del = st.columns([10, 1])
                c_fn.markdown(f"`{fn}`")
                if c_del.button("🗑", key=f"del_resp_{fn}", help="Remove"):
                    os.remove(os.path.join(resp_dir, fn))
                    st.rerun()
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <div class="empty-msg">No documents uploaded yet.</div>
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

    # ── Key metrics ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tender type",  (report.get("tender_type") or "—").upper())
    c2.metric("Deadline",      report.get("submission_deadline") or "—")
    c3.metric("Est. value",    report.get("estimated_value_eur") or "—")
    c4.metric("Authority",    (report.get("contracting_authority") or "—")[:30])

    # ── Executive Summary ──────────────────────────────────────────
    st.markdown('<div class="section-heading">Executive Summary</div>', unsafe_allow_html=True)
    for line in report.get("executive_summary", []):
        st.markdown(f"- {line}")

    # ── Go / No-Go verdict ─────────────────────────────────────────
    # (already rendered above as banner — showstoppers follow immediately)
    showstoppers = report.get("showstoppers", [])
    if showstoppers:
        st.markdown(
            f'<div class="section-heading section-heading-orange">Showstoppers ({len(showstoppers)})</div>',
            unsafe_allow_html=True,
        )
        for ss in showstoppers:
            st.markdown(f"""
<div class="ss-card">
  <div class="ss-id">{ss.get("id", "SS")}</div>
  <div class="ss-desc">{ss.get("description", "")}</div>
  <div class="ss-evidence">Evidence: {ss.get("evidence", "—")} · Impact: {ss.get("impact", "—")}</div>
</div>""", unsafe_allow_html=True)

    # ── Key Deadlines & Milestones ─────────────────────────────────
    deadlines = report.get("deadlines", [])
    st.markdown('<div class="section-heading">Key Deadlines & Milestones</div>', unsafe_allow_html=True)
    if deadlines:
        for d in deadlines:
            st.markdown(f"- **{d.get('when','?')}** — {d.get('milestone','')}  \n  *{d.get('evidence','')}*")
    else:
        st.markdown('<p class="info-box">No specific dates identified in the document.</p>', unsafe_allow_html=True)

    # ── Requirements ───────────────────────────────────────────────
    reqs = report.get("requirements", {})
    st.markdown('<div class="section-heading">Requirements & Constraints</div>', unsafe_allow_html=True)
    if reqs:
        for key, items in reqs.items():
            if items:
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                for item in items:
                    st.markdown(f"- {item}")
    else:
        st.markdown('<p class="info-box">No requirements extracted.</p>', unsafe_allow_html=True)

    # ── Deliverables ───────────────────────────────────────────────
    deliverables = report.get("deliverables", [])
    st.markdown('<div class="section-heading">Deliverables to Prepare</div>', unsafe_allow_html=True)
    if deliverables:
        for item in deliverables:
            st.markdown(f"- {item}")
    else:
        st.markdown('<p class="info-box">No deliverables listed.</p>', unsafe_allow_html=True)

    # ── Risk Register ──────────────────────────────────────────────
    risks = report.get("risks", [])
    st.markdown('<div class="section-heading">Risk Register</div>', unsafe_allow_html=True)
    if risks:
        df = pd.DataFrame(risks)
        ordered = [c for c in ["id","risk","category","probability","impact","score","evidence","mitigation"] if c in df.columns]
        st.dataframe(df[ordered].sort_values("score", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.markdown('<p class="info-box">No risks identified.</p>', unsafe_allow_html=True)

    # ── Open Questions ─────────────────────────────────────────────
    open_qs = report.get("open_questions", [])
    if open_qs:
        st.markdown('<div class="section-heading">Open Questions</div>', unsafe_allow_html=True)
        for q in open_qs:
            st.markdown(f"- {q}")

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
