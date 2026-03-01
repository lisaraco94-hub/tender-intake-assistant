"""
Tender Intake Assistant — Pre-Bid Screening
AI-powered analysis for TLA/IVD clinical laboratory tenders.
"""

import inspect
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.extractors import extract_from_file, parse_bid_response_excel, SUPPORTED_EXTENSIONS
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

# ─── Absolute base dir (resolves regardless of CWD) ──────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Library persistence ─────────────────────────────────────────
LIBRARY_PATH = os.path.join(_APP_DIR, "assets", "tender_library.json")


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


RISK_FACTORS_PATH = os.path.join(_APP_DIR, "assets", "risk_factors.json")


def save_risk_factors(rf: dict):
    with open(RISK_FACTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(rf, f, ensure_ascii=False, indent=2)


def _migrate_risk_register(rf: dict) -> tuple[dict, bool]:
    """
    One-time migration: flatten legacy high_risks / medium_risks into a single
    risk_factors list with level + score fields. Returns (rf, was_migrated).
    """
    rr = rf.get("risk_register", {})
    migrated = False
    if "high_risks" in rr or "medium_risks" in rr:
        unified = rr.setdefault("risk_factors", [])
        for item in rr.pop("high_risks", []):
            item.setdefault("level", "High")
            item.setdefault("score", 75)
            unified.append(item)
        for item in rr.pop("medium_risks", []):
            item.setdefault("level", "Medium")
            item.setdefault("score", 50)
            unified.append(item)
        rf["risk_register"] = rr
        migrated = True
    return rf, migrated


def _load_knowledge_context(max_chars_per_file: int = 20_000, max_total: int = 80_000) -> str:
    """Load past bid response documents from the knowledge base.

    Excel files (.xlsx/.xls) are parsed with the dedicated compliance-matrix
    parser so that Y/N/partially answers are correctly categorised and the AI
    receives a structured summary rather than raw tab-separated cell dumps.
    Other formats are read as plain text.
    """
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
                ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
                if ext in ("xlsx", "xls"):
                    text = parse_bid_response_excel(file_bytes, fn)
                else:
                    pages = extract_from_file(file_bytes, fn)
                    text = "\n".join(pages)
                text = text[:max_chars_per_file]
                parts.append(f"=== {label}: {fn} ===\n{text}")
                total += len(text)
            except Exception:
                pass
    return "\n\n".join(parts)


def _ai_format_risk(concept: str, entry_type: str, rf: dict, level: str = "Medium") -> dict:
    """Convert a plain-language risk description into a structured JSON entry using GPT-4o."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    if entry_type == "showstopper":
        existing = rf.get("risk_register", {}).get("showstoppers", [])
        prefix = "SS"
    else:
        existing = rf.get("risk_register", {}).get("risk_factors", [])
        prefix = "RF"

    max_n = 0
    for e in existing:
        try:
            max_n = max(max_n, int(e.get("id", "0").split("-")[-1]))
        except Exception:
            pass
    next_id = f"{prefix}-{max_n + 1:02d}"

    level_score = {"Low": 25, "Medium": 50, "High": 75}.get(level, 50)

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
            f'"category": "e.g. Technical / Commercial / Legal / Operational", '
            f'"level": "{level}", '
            f'"score": {level_score}}}'
        )
        context = f"This is a {level.upper()} RISK factor."

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


def _generate_library_description(report: dict) -> str:
    """Generate a concise AI summary of the tender type for the library card."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        summary_text = "\n".join(report.get("executive_summary", []))
        reqs = report.get("requirements", {})
        reqs_text = ""
        for cat, items in reqs.items():
            reqs_text += f"{cat}: " + "; ".join(str(i) for i in items[:3]) + "\n"
        risks_text = ", ".join(r.get("risk","") for r in report.get("risks",[])[:5])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": (
                    "You are a tender analyst for Inpeco (Total Laboratory Automation). "
                    "Write a single concise sentence (max 30 words) describing this tender. "
                    "Immediately flag if it is a TLA (Total Laboratory Automation) project or partial. "
                    "Mention if multi-floor, multi-lab, multi-site. "
                    "Mention specific specialties (e.g. microbiology, molecular, blood bank) if involved. "
                    "Output only the sentence, no prefix."
                ),
            }, {
                "role": "user",
                "content": f"Summary:\n{summary_text}\n\nRequirements:\n{reqs_text}\nTop risks: {risks_text}",
            }],
            temperature=0.2,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return (report.get("executive_summary") or [""])[0][:120]


def _lookup_location_online(authority_name: str) -> tuple[str, str]:
    """Use GPT to infer city and country for a contracting authority if not found in docs."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": (
                    "You are a geography expert. Given a hospital or institution name, "
                    "return the most likely city and country. "
                    'Return ONLY valid JSON: {"city": "...", "country": "..."} '
                    'If unknown, use empty strings.'
                ),
            }, {
                "role": "user",
                "content": f"Institution: {authority_name}",
            }],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=60,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("city", ""), data.get("country", "")
    except Exception:
        return "", ""


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
if "lib_selected" not in st.session_state:
    st.session_state.lib_selected = None   # index into load_library() for drill-down

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

/* Hide Streamlit's native header bar (replaced by our custom nav) */
header[data-testid="stHeader"] {{
    display: none !important;
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

/* ── Feature cards — equal-height columns ── */
[data-testid="stHorizontalBlock"] {{
    align-items: stretch !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    display: flex !important;
    flex-direction: column !important;
}}
/* Stretch every intermediate wrapper inside the column */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] [data-testid="stMarkdownContainer"] {{
    flex: 1 !important;
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
    display: flex;
    flex-direction: column;
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
    flex: 1;
}}
.feat-badge {{
    display: inline-block;
    background: {PRIMARY}22;
    color: {PRIMARY};
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: auto;
    align-self: center;
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
    color: rgba(255,255,255,0.85);
    line-height: 1.5;
}}

/* ── Library rows ── */
.lib-row {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.7rem;
    border-left: 5px solid {PRIMARY};
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}}
.lib-row-nogo {{ border-left-color: #e74c3c; }}
.lib-row-mit  {{ border-left-color: {ORANGE}; }}
.lib-client-title {{
    font-size: 1.05rem;
    font-weight: 800;
    color: {NAVY};
    letter-spacing: 0.01em;
    margin-bottom: 0.18rem;
}}
.lib-meta     {{ font-size: 0.74rem; color: #7a96b0; margin-top: 0.1rem; }}
.lib-summary  {{ font-size: 0.8rem; color: #3a5a7a; margin-top: 0.4rem; line-height: 1.55; }}
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

/* ── Library: clickable row ── */
.lib-row {{ cursor: pointer; transition: box-shadow 0.15s, transform 0.12s; }}
.lib-row:hover {{ box-shadow: 0 6px 20px rgba(0,174,239,0.18); transform: translateY(-1px); }}

/* ── Portfolio Insights panel ── */
.insights-panel {{
    background: linear-gradient(135deg, {NAVY} 0%, #005a96 100%);
    border-radius: 14px;
    padding: 1.5rem 1.8rem 1.6rem;
    margin-bottom: 1.8rem;
    color: white;
}}
.insights-title {{
    font-size: 1.05rem;
    font-weight: 800;
    color: {PRIMARY};
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
}}
.insight-tag {{
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(0,174,239,0.4);
    border-radius: 20px;
    padding: 0.22rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #e0f4ff;
    margin: 0.2rem 0.2rem 0.2rem 0;
    white-space: nowrap;
}}
.insight-tag-hot {{
    background: rgba(247,148,29,0.25);
    border-color: {ORANGE};
    color: #ffe0b2;
}}

/* ── Report page: white container ── */
.report-page {{
    background: {WHITE};
    border-radius: 18px;
    padding: 2.2rem 2.5rem 2.5rem;
    margin-top: 1rem;
    box-shadow: 0 4px 24px rgba(0,56,101,0.10);
}}
.rpt-h1 {{
    font-size: 1.15rem;
    font-weight: 800;
    color: {PRIMARY};
    border-left: 4px solid {PRIMARY};
    padding-left: 0.75rem;
    margin: 1.8rem 0 0.9rem;
    letter-spacing: 0.01em;
}}
.rpt-h1-orange {{
    color: {ORANGE};
    border-left-color: {ORANGE};
}}
.rpt-verdict {{
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.4rem;
    border-left-width: 5px;
    border-left-style: solid;
}}
.rpt-verdict-go     {{ background: #f0fdf6; border-left-color: #2ecc71; }}
.rpt-verdict-go-mit {{ background: #fffbf0; border-left-color: {ORANGE}; }}
.rpt-verdict-nogo   {{ background: #fff5f5; border-left-color: #e74c3c; }}
.rpt-verdict-label  {{ font-size: 1.55rem; font-weight: 800; color: {NAVY}; letter-spacing: 0.05em; }}
.rpt-verdict-score  {{ font-size: 0.85rem; color: #6a88aa; margin-top: 0.3rem; }}
.rpt-verdict-rationale {{ font-size: 0.88rem; color: #3a5570; margin-top: 0.6rem; font-style: italic; line-height: 1.55; }}
.rpt-ss-card {{
    background: #fff5f0;
    border-left: 4px solid {ORANGE};
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.6rem;
}}
.rpt-ss-id  {{ font-size: 0.7rem; font-weight: 700; color: {ORANGE}; letter-spacing: 0.07em; text-transform: uppercase; }}
.rpt-ss-desc {{ font-size: 0.92rem; font-weight: 700; color: {NAVY}; margin: 0.2rem 0 0.3rem; }}
.rpt-ss-ref  {{ font-size: 0.78rem; color: {PRIMARY}; font-weight: 600; }}
.rpt-ss-ev   {{ font-size: 0.78rem; color: #5a7a9a; line-height: 1.5; margin-top: 0.15rem; }}
.rpt-risk-card {{
    background: #f8fbff;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid {PRIMARY};
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}}
.rpt-risk-card-high   {{ border-left-color: #e74c3c; background: #fff8f8; }}
.rpt-risk-card-medium {{ border-left-color: {ORANGE}; background: #fffbf5; }}
.rpt-risk-card-low    {{ border-left-color: #2ecc71; background: #f5fdf8; }}
.rpt-risk-level {{
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
    white-space: nowrap;
    flex-shrink: 0;
}}
.level-high   {{ background: #fde8e8; color: #c0392b; }}
.level-medium {{ background: #fef0c7; color: #a06000; }}
.level-low    {{ background: #d4f5e5; color: #1a7a48; }}
.rpt-risk-score {{
    font-size: 1.1rem;
    font-weight: 800;
    color: {NAVY};
    min-width: 2.5rem;
    text-align: center;
    flex-shrink: 0;
}}
.rpt-risk-score-label {{ font-size: 0.6rem; color: #8aa0b8; font-weight: 500; }}
.rpt-risk-body {{ flex: 1; min-width: 0; }}
.rpt-risk-name {{ font-size: 0.9rem; font-weight: 700; color: {NAVY}; margin-bottom: 0.18rem; }}
.rpt-risk-cat  {{ font-size: 0.7rem; color: {PRIMARY}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
.rpt-risk-ref  {{ font-size: 0.75rem; color: {PRIMARY}; font-weight: 600; margin-top: 0.3rem; }}
.rpt-risk-ev   {{ font-size: 0.75rem; color: #5a7a9a; font-style: italic; margin-top: 0.15rem; line-height: 1.4; }}
.rpt-risk-mit  {{ font-size: 0.75rem; color: #3a6a4a; margin-top: 0.2rem; }}
.rpt-bullet-item {{
    font-size: 0.87rem;
    color: #2a3a4a;
    padding: 0.3rem 0 0.3rem 1rem;
    border-bottom: 1px solid #f0f4f8;
    line-height: 1.5;
}}
.rpt-deadline-row {{
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0f4f8;
    font-size: 0.85rem;
    color: #2a3a4a;
}}
.rpt-deadline-when {{ font-weight: 700; color: {NAVY}; min-width: 6rem; flex-shrink: 0; }}
.rpt-deadline-ev   {{ font-size: 0.75rem; color: #7a96b0; font-style: italic; }}

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
    color: rgba(255,255,255,0.82) !important;
}}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectSlider"] label,
[data-testid="stFileUploader"] label,
[data-testid="stSelectbox"] label {{
    color: rgba(255,255,255,0.82) !important;
}}
/* Keep drop zone inner text dark (it sits on a white box) */
[data-testid="stFileDropzone"] span,
[data-testid="stFileDropzone"] small,
[data-testid="stFileDropzone"] p,
[data-testid="stFileDropzone"] > div {{
    color: #2a4a6a !important;
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
.empty-msg  {{ font-size: 0.9rem; color: rgba(255,255,255,0.88); margin-top: 0.8rem; line-height: 1.7; }}

/* ── Analysis depth: scoped intensity blocks (marker = .depth-widget-marker) ── */
.depth-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: rgba(255,255,255,0.88);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.3rem;
}}
/* Scope via sibling of the marker container */
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    gap: 0.45rem !important;
    margin-top: 0.2rem !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label {{
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    padding: 0.55rem 0.5rem 0.45rem !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.1) !important;
    border: 2px solid rgba(255,255,255,0.18) !important;
    cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s !important;
    gap: 0.32rem !important;
    white-space: nowrap !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label [data-baseweb="radio"] {{
    display: none !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label p {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: rgba(255,255,255,0.92) !important;
    margin: 0 !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
}}
/* Intensity bars: low=light orange, medium=orange, high=dark red */
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label::before {{
    content: '' !important;
    display: block !important;
    width: 100% !important;
    border-radius: 3px !important;
    flex-shrink: 0 !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(1)::before {{
    height: 4px !important;
    background: rgba(247,148,29,0.45) !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(2)::before {{
    height: 8px !important;
    background: {ORANGE} !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(3)::before {{
    height: 12px !important;
    background: linear-gradient(135deg, {ORANGE} 0%, #c0392b 100%) !important;
}}
/* Selected: highlight border + filled navy dot below text */
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
    border-color: rgba(255,255,255,0.75) !important;
    background: rgba(255,255,255,0.22) !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label::after {{
    content: '' !important;
    display: block !important;
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: transparent !important;
    flex-shrink: 0 !important;
}}
[data-testid="stMarkdownContainer"]:has(.depth-widget-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::after {{
    background: {NAVY} !important;
}}

/* ── KB type selector: vertical checklist (marker = .kb-type-marker) ── */
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [data-testid="stWidgetLabel"] {{
    display: none !important;
}}
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] {{
    display: flex !important;
    flex-direction: column !important;
    gap: 0.35rem !important;
    margin-top: 0.3rem !important;
}}
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label {{
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 0.65rem !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.08) !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    cursor: pointer !important;
    transition: background 0.15s, border-color 0.15s !important;
}}
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label [data-baseweb="radio"] {{
    display: none !important;
}}
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label p {{
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.92) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}}
/* Circle indicator via ::before */
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label::before {{
    content: '' !important;
    display: block !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    border: 2px solid rgba(255,255,255,0.45) !important;
    background: transparent !important;
    flex-shrink: 0 !important;
    transition: background 0.15s, border-color 0.15s !important;
}}
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::before {{
    border-color: white !important;
    background: white !important;
}}
[data-testid="stMarkdownContainer"]:has(.kb-type-marker) ~ [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
    background: rgba(255,255,255,0.18) !important;
    border-color: rgba(255,255,255,0.5) !important;
}}
[data-testid="stPlotlyChart"] {{
    flex: none !important;
    height: 1600px !important;
    min-height: 1600px !important;
}}
[data-testid="stPlotlyChart"] iframe {{
    height: 1600px !important;
    min-height: 1600px !important;
}}
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
              Add Inpeco-specific showstoppers and risks by typing
              in plain text. Upload past bid responses
              to sharpen analysis accuracy over time.
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
    col_key, col_depth = st.columns([3, 2], gap="large")
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
        st.markdown('<div class="depth-widget-marker"><div class="depth-label">Analysis depth</div></div>', unsafe_allow_html=True)
        detail = st.radio(
            "depth_sel",
            options=["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(st.session_state.detail),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.detail = detail
        st.caption({
            "Low":    "~1–2 min · showstoppers only",
            "Medium": "~2–4 min · risks + requirements",
            "High":   "~4–8 min · full analysis",
        }[detail])

        st.markdown(
            '<div style="margin-top:0.7rem;font-size:0.78rem;color:rgba(255,255,255,0.75);'
            'font-weight:600;margin-bottom:0.15rem;">Consensus runs</div>',
            unsafe_allow_html=True,
        )
        consensus_runs = st.radio(
            "consensus_runs_sel",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1× Fast", 2: "2× Balanced", 3: "3× Consensus"}[x],
            index=st.session_state.get("consensus_runs", 1) - 1,
            horizontal=True,
            label_visibility="collapsed",
            key="consensus_runs_radio",
        )
        st.session_state.consensus_runs = consensus_runs
        st.caption({
            1: "Single pass",
            2: "2 independent runs merged",
            3: "3 independent runs merged — most stable",
        }[consensus_runs])

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
            skipped: list[str] = []
            with st.spinner("Reading files…"):
                for uf in uploaded_files:
                    try:
                        pages = extract_from_file(uf.read(), uf.name)
                    except Exception as _e:
                        skipped.append(f"{uf.name} ({_e})")
                        continue
                    # extract_from_file returns a warning string on soft errors
                    if len(pages) == 1 and pages[0].startswith("("):
                        skipped.append(f"{uf.name} — {pages[0].strip('()')}")
                        continue
                    all_pages.append(f"=== FILE: {uf.name} ===")
                    all_pages.extend(pages)
            if skipped:
                st.warning(
                    "⚠️ The following file(s) could not be read and were skipped:\n\n"
                    + "\n".join(f"• {s}" for s in skipped)
                )
            if not all_pages:
                st.error("No readable content found in the uploaded files. Please check the file formats.")
                st.stop()

            knowledge_ctx = _load_knowledge_context()
            n_kb = len([
                fn
                for folder in (
                    "assets/knowledge/responses",
                    "assets/knowledge/won",
                    "assets/knowledge/lost",
                )
                if os.path.exists(folder)
                for fn in os.listdir(folder)
            ])
            runs = st.session_state.get("consensus_runs", 1)
            spinner_msg = (
                f"Analysing {len(uploaded_files)} file(s) with GPT-4o [{detail}]"
                + (f" · {n_kb} KB doc(s) in KB" if n_kb else "")
                + (f" · {runs}× consensus" if runs > 1 else "")
                + "…"
            )
            with st.spinner(spinner_msg):
                try:
                    _sig = inspect.signature(build_prebid_report)
                    _kw: dict = dict(
                        risk_factors=risk_factors,
                        detail=detail,
                    )
                    if "knowledge_context" in _sig.parameters:
                        _kw["knowledge_context"] = knowledge_ctx
                    if "runs" in _sig.parameters:
                        _kw["runs"] = runs
                    if "knowledge_context" not in _sig.parameters and knowledge_ctx:
                        # Legacy pipeline without knowledge_context param — prepend as page
                        all_pages = [f"=== COMPANY KNOWLEDGE BASE ===\n{knowledge_ctx}"] + all_pages
                    report = build_prebid_report(all_pages, **_kw)
                    st.session_state.report = report
                    st.session_state.run_done = True

                    # Resolve city/country — fall back to online lookup
                    city    = report.get("city", "").strip()
                    country = report.get("country", "").strip()
                    authority = report.get("contracting_authority", "") or ""
                    if (not city or not country) and authority:
                        with st.spinner("Locating contracting authority…"):
                            c2, co2 = _lookup_location_online(authority)
                            city    = city    or c2
                            country = country or co2

                    # Build display title: Client - City (Country)
                    loc_parts = []
                    if city:    loc_parts.append(city)
                    if country: loc_parts.append(f"({country})")
                    loc_str = " ".join(loc_parts) if loc_parts else ""
                    client_label = authority or report.get("tender_title") or "—"
                    display_title = f"{client_label} – {loc_str}" if loc_str else client_label

                    # AI-generated description
                    with st.spinner("Generating summary…"):
                        ai_desc = _generate_library_description(report)

                    save_to_library({
                        "date":          datetime.now().strftime("%Y-%m-%d"),
                        "display_title": display_title,
                        "title":         report.get("tender_title") or "—",
                        "client":        authority or "—",
                        "city":          city or "—",
                        "country":       country or "—",
                        "verdict":       report.get("go_nogo", {}).get("recommendation", "—"),
                        "score":         report.get("go_nogo", {}).get("score", 0),
                        "summary":       ai_desc,
                        "files":         [f.name for f in uploaded_files],
                        "report":        report,   # full JSON for drill-down + Word export
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    if st.session_state.run_done and st.session_state.report:
        _render_report(st.session_state.report)


# ─── LIBRARY helpers ──────────────────────────────────────────────

def _portfolio_insights(lib: list):
    """Render the Portfolio Risk Insights panel from all stored reports."""
    from collections import Counter
    import plotly.express as px
    import plotly.graph_objects as go

    entries_with_report = [e for e in lib if e.get("report")]
    if not entries_with_report:
        return  # nothing to aggregate yet

    import re as _re

    # ── Named-entity patterns per category: (regex, canonical_label) ──
    # Each list is matched against the specific report field most likely to contain it.
    _CERTS = [
        (r"ISO[\s\-]?13485", "ISO 13485"),
        (r"ISO[\s\-]?9001", "ISO 9001"),
        (r"ISO[\s\-]?15189", "ISO 15189"),
        (r"ISO[\s\-]?27001", "ISO 27001"),
        (r"ISO[\s\-]?62443", "ISO 62443"),
        (r"ISO[\s\-]?14971", "ISO 14971"),
        (r"CE[\s\-]?mark(?:ing)?", "CE Marking"),
        (r"\bIVDR\b", "IVDR"),
        (r"\bMDR\b", "MDR"),
        (r"\bFDA\b", "FDA"),
        (r"\bcGMP\b|\bGMP\b", "GMP"),
        (r"\bGDPR\b", "GDPR"),
        (r"\bIQ[/\\]OQ[/\\]PQ\b|\bIQ[/\\]OQ\b|\bOQ[/\\]PQ\b", "IQ/OQ/PQ"),
        (r"\bCAP\b", "CAP"),
    ]
    # Analyzer instruments: specific model names take priority over brand labels.
    # Because we use a set(), both a brand tag and a model tag can coexist for the same tender.
    _ANALYZERS = [
        # Roche / Cobas — models
        (r"\bCobas\s*8100\b", "Cobas 8100"),
        (r"\bCobas\s*Pro\b", "Cobas Pro"),
        (r"\bCobas\s*6800\b", "Cobas 6800"),
        (r"\bCobas\s*8800\b", "Cobas 8800"),
        (r"\bCobas\s*[ec]\s*\d{3,4}\b", "Cobas e/c-series"),
        (r"\bCobas\s*p\s*\d+\b|\bCobas\s*infinity\b", "Cobas p-series"),
        (r"\bRoche\b|\bCobas\b", "Roche / Cobas"),
        # Abbott — models
        (r"\bAlinity\s*[cis]\b|\bAlinity\b", "Alinity"),
        (r"\bArchitect\s*[ci]\s*\d+\b|\bArchitect\b", "Architect"),
        (r"\bAbbott\b", "Abbott"),
        # Siemens — models
        (r"\bAtellica\s*(?:Solution|CI|IM|CH)\b|\bAtellica\b", "Atellica"),
        (r"\bAdvia\s*\d+\b|\bAdvia\b", "Advia"),
        (r"\bBN\s*Pro\s*Spec\b|\bBNPro\b", "BN ProSpec"),
        (r"\bSiemens\b|\bDimension\b", "Siemens"),
        # Beckman Coulter — models
        (r"\bDxC\s*\d+\b", "DxC"),
        (r"\bAccess\s*2\b|\bAccess\s*\w+\b", "Access"),
        (r"\bBeckman\s*Coulter\b|\bUniCel\b", "Beckman Coulter"),
        # Sysmex — models
        (r"\bXN[\s\-]?\d{3,4}\b", "Sysmex XN"),
        (r"\bXE[\s\-]?\d+\b|\bXS[\s\-]?\d+\b", "Sysmex XE/XS"),
        (r"\bSysmex\b", "Sysmex"),
        # Others
        (r"\bMindray\b", "Mindray"),
        (r"\bHoriba\b", "Horiba"),
        (r"\bbio\s*M[eé]rieux\b|\bVITEK\b", "bioMérieux"),
        (r"\bTosoh\b", "Tosoh"),
        (r"\bOrtho\s*Clinical\b|\bVitros\b", "Ortho Clinical"),
        (r"\bSebia\b", "Sebia"),
        (r"\bWerfen\b|\bILab\b|\bStago\b", "Werfen / Stago"),
        (r"\bDiaSorin\b|\bLIAISON\b", "DiaSorin"),
        (r"\bSnibe\b|\bMaglumi\b", "Snibe"),
        (r"\bThermo\s*Fisher\b|\bBRAHMS\b", "ThermoFisher"),
        (r"\bHologic\b", "Hologic"),
        (r"\bSarstedt\b", "Sarstedt"),
    ]
    _LIS = [
        (r"\bDedalus\b|\bDHE\b", "Dedalus"),
        (r"\bTrakCare\b|\bInterSystems\b|\bHealthShare\b", "InterSystems"),
        (r"\bEpic\b", "Epic"),
        (r"\bCerner\b", "Cerner"),
        (r"\bMeditech\b|\bMEDITECH\b", "Meditech"),
        (r"\bMolis\b|\bHexalis\b", "Molis"),
        (r"\bSinfonia\b", "Sinfonia"),
        (r"\bLabVantage\b", "LabVantage"),
        (r"\bSunquest\b", "Sunquest"),
        (r"\bRemisol\b", "Remisol"),
        (r"\bCliniSys\b|\bWinPath\b", "CliniSys"),
        (r"\bHL7\b", "HL7"),
        (r"\bASTM\b", "ASTM"),
        (r"\bFHIR\b", "FHIR"),
        (r"\bDICOM\b", "DICOM"),
    ]
    # Lab specialties — extracted from analyzer_connectivity
    _SPECIALTIES = [
        (r"\bchimica\s*clinica\b|\bclinical\s*chemi(?:stry|e)\b|\bKlinische\s*Chemie\b", "Chimica clinica"),
        (r"\bimmuno(?:assay|metria|metric|chimica)\b", "Immunoassay"),
        (r"\bematologia\b|\bh[ae]matology\b", "Ematologia"),
        (r"\bcoagulazione\b|\bcoagulation\b|\bemostasi\b|\bh[ae]mostasis\b", "Coagulazione"),
        (r"\burin(?:analisi|alisi|analysis)\b|\besame\s*urine\b", "Urinanalisi"),
        (r"\bmicrobiologia\b|\bmicrobiology\b", "Microbiologia"),
        (r"\bbiologia\s*molecolare\b|\bmolecular\s*(?:biology|diagnostics)\b|\breal[\s\-]?time\s*PCR\b|\bNGS\b|\b\bPCR\b", "Biologia molecolare"),
        (r"\bemogasanalisi\b|\bblood\s*gas\b|\bEGA\b|\bgasometria\b", "Emogasanalisi"),
        (r"\btossicologia\b|\btoxicology\b|\bdrug\s*testing\b|\bTDM\b", "Tossicologia"),
        (r"\bsierologia\b|\bserology\b", "Sierologia"),
        (r"\belettroforesi\b|\belectrophoresis\b|\bSPEP\b|\bIPEP\b", "Elettroforesi"),
        (r"\bcitofluorimetria\b|\bflow\s*cytometry\b", "Citofluorimetria"),
        (r"\bpunto\s*di\s*cura\b|\bpoint[\s\-]of[\s\-]care\b|\bPOCT\b", "POCT"),
    ]
    # Infrastructure & operational requirements — extracted from scope + space + commercial fields
    _INFRA = [
        # BIM / digital formats
        (r"\bBIM\b|\bBuilding\s*Information\s*Model(?:l?ing)?\b", "BIM"),
        (r"\bIFC\b|\bIndustry\s*Foundation\s*Class(?:es)?\b", "IFC format"),
        (r"\bRevit\b|\bAutoCAD\b|\bCAD\b", "CAD / Revit"),
        (r"\bdigital\s*twin\b|\bgemello\s*digitale\b", "Digital twin"),
        # Civil / building works
        (r"\blavori\s*edili\b|\bopere\s*civili\b|\bedilizia\b|\bcivil\s*works\b|\bBauarbeiten\b", "Lavori edili"),
        (r"\bdemoliz(?:ione|ioni)\b|\bdismissione\b|\brimozione\s*(?:arredi|attrezzature)\b|\bsmantellamento\b", "Demolizione/smantellamento"),
        (r"\bimpianti?\s*elettri(?:ci|co)\b|\belectrical\s*works\b|\bcablaggi\b", "Impianti elettrici"),
        (r"\bimpianti?\s*(?:idraulici?|idrico|pneumatic[oi]|aria\s*compressa)\b", "Impianti idraulici/pneumatici"),
        (r"\bpaviment(?:azione|o\s+sopraelevato)\b|\braised\s*floor\b|\bfalso\s*pavimento\b", "Pavimentazione"),
        # On-site service / support model
        (r"\bpresidio\s*(?:fisso|continuo|permanente|giornaliero|quotidiano|h24)\b|\bon[\s\-]?site\s*engineer\b|\btechnicien\s*(?:d[eé]di[eé]|sur\s*site)\b", "Presidio on-site"),
        (r"\btecnico\s*dedicato\b|\bdedicated\s*(?:engineer|technician|service)\b|\bingegnere\s*dedicato\b", "Tecnico dedicato"),
        (r"\b24[/\\]7\b|\bH24\b|\b24\s*ore\s*su\s*24\b|\bsette\s*giorni\s*su\s*sette\b", "Supporto 24/7"),
        (r"\breperibilit[àa]\b|\bon[\s\-]?call\b|\bguardia\s*attiva\b", "Reperibilità h24"),
        # Remote / digital services
        (r"\bremote\s*monitoring\b|\bmonitoraggio\s*remoto\b|\bteleassistenza\b|\btelemanutenzione\b", "Monitoraggio remoto"),
        (r"\bmagazzino\s*ricambi\b|\bspare\s*parts\s*on[\s\-]?site\b|\bstock\s*ricambi\b", "Ricambi in loco"),
    ]

    def _match_labels(patterns, text):
        found = set()
        for pattern, label in patterns:
            if _re.search(pattern, text, _re.IGNORECASE):
                found.add(label)
        return found

    # Country name → ISO-3 mapping
    _ISO3 = {
        "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA",
        "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS",
        "Austria": "AUT", "Azerbaijan": "AZE", "Bahrain": "BHR",
        "Bangladesh": "BGD", "Belarus": "BLR", "Belgium": "BEL",
        "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH", "Brazil": "BRA",
        "Bulgaria": "BGR", "Canada": "CAN", "Chile": "CHL",
        "China": "CHN", "Colombia": "COL", "Croatia": "HRV",
        "Cyprus": "CYP", "Czech Republic": "CZE", "Czechia": "CZE",
        "Denmark": "DNK", "Ecuador": "ECU", "Egypt": "EGY",
        "Estonia": "EST", "Ethiopia": "ETH", "Finland": "FIN",
        "France": "FRA", "Georgia": "GEO", "Germany": "DEU",
        "Ghana": "GHA", "Greece": "GRC", "Hungary": "HUN",
        "India": "IND", "Indonesia": "IDN", "Iran": "IRN",
        "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
        "Italy": "ITA", "Japan": "JPN", "Jordan": "JOR",
        "Kazakhstan": "KAZ", "Kenya": "KEN", "Kosovo": "XKX",
        "Kuwait": "KWT", "Latvia": "LVA", "Lebanon": "LBN",
        "Libya": "LBY", "Lithuania": "LTU", "Luxembourg": "LUX",
        "Malaysia": "MYS", "Malta": "MLT", "Mexico": "MEX",
        "Moldova": "MDA", "Montenegro": "MNE", "Morocco": "MAR",
        "Netherlands": "NLD", "New Zealand": "NZL", "Nigeria": "NGA",
        "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN",
        "Pakistan": "PAK", "Panama": "PAN", "Peru": "PER",
        "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
        "Qatar": "QAT", "Romania": "ROU", "Russia": "RUS",
        "Saudi Arabia": "SAU", "Serbia": "SRB", "Singapore": "SGP",
        "Slovakia": "SVK", "Slovenia": "SVN", "South Africa": "ZAF",
        "South Korea": "KOR", "Spain": "ESP", "Sweden": "SWE",
        "Switzerland": "CHE", "Syria": "SYR", "Taiwan": "TWN",
        "Tanzania": "TZA", "Thailand": "THA", "Tunisia": "TUN",
        "Turkey": "TUR", "Türkiye": "TUR", "Uganda": "UGA",
        "Ukraine": "UKR", "United Arab Emirates": "ARE",
        "United Kingdom": "GBR", "UK": "GBR",
        "USA": "USA", "United States": "USA", "United States of America": "USA",
        "Uzbekistan": "UZB", "Venezuela": "VEN", "Vietnam": "VNM",
        "Yemen": "YEM", "Zimbabwe": "ZWE",
    }
    _ISO3_TO_NAME = {v: k for k, v in _ISO3.items()}

    # ── Collect per-entry data ──
    countries_entries: dict = {}
    entry_tags: list = []

    for entry in entries_with_report:
        r = entry["report"]
        country = (entry.get("country") or r.get("country") or "").strip()
        if country:
            countries_entries.setdefault(country, []).append(entry)
        reqs = r.get("requirements", {})
        cert_text    = " ".join(reqs.get("qualification_and_compliance", []))
        anlzr_text   = " ".join(reqs.get("analyzer_connectivity", []))
        lis_text     = " ".join(reqs.get("it_and_middleware", []))
        # specialties live in analyzer_connectivity too (AI describes what's needed there)
        spec_text    = anlzr_text
        # infra/operational requirements span scope, space, and commercial fields
        infra_text   = " ".join(
            reqs.get("scope_and_responsibility", [])
            + reqs.get("space_and_facility", [])
            + reqs.get("commercial_conditions", [])
        )
        tags = {
            "cert":      _match_labels(_CERTS,       cert_text),
            "analyzer":  _match_labels(_ANALYZERS,   anlzr_text),
            "lis":       _match_labels(_LIS,          lis_text),
            "specialty": _match_labels(_SPECIALTIES,  spec_text),
            "infra":     _match_labels(_INFRA,        infra_text),
        }
        entry_tags.append((entry, tags))

    def _tally(pairs):
        cert_c = Counter(); anlzr_c = Counter(); lis_c = Counter()
        spec_c = Counter(); infra_c = Counter()
        for _, t in pairs:
            for lbl in t["cert"]:      cert_c[lbl]  += 1
            for lbl in t["analyzer"]:  anlzr_c[lbl] += 1
            for lbl in t["lis"]:       lis_c[lbl]   += 1
            for lbl in t["specialty"]: spec_c[lbl]  += 1
            for lbl in t["infra"]:     infra_c[lbl] += 1
        return cert_c, anlzr_c, lis_c, spec_c, infra_c

    cert_all, anlzr_all, lis_all, spec_all, infra_all = _tally(entry_tags)

    if not (countries_entries or any([cert_all, anlzr_all, lis_all, spec_all, infra_all])):
        return

    # ── Build choropleth data ──
    map_rows = []
    for country, entries in countries_entries.items():
        iso3 = _ISO3.get(country, "")
        if iso3:
            map_rows.append({"country": country, "iso3": iso3, "count": len(entries)})

    # ── Render ──
    st.markdown("""
    <div class="insights-panel">
      <div class="insights-title">📊 Portfolio Risk Insights</div>
    """, unsafe_allow_html=True)

    if "map_selected_iso3" not in st.session_state:
        st.session_state["map_selected_iso3"] = set()

    selected_names: list = []
    cols = st.columns([3, 2])

    with cols[0]:
        st.markdown("**Tenders by country** — click to filter tags")

        if map_rows:
            df_map = pd.DataFrame(map_rows)

            _sel_iso3 = st.session_state["map_selected_iso3"]
            df_unsel = df_map[~df_map["iso3"].isin(_sel_iso3)]
            df_sel   = df_map[df_map["iso3"].isin(_sel_iso3)]

            fig = px.choropleth(
                df_unsel,
                locations="iso3",
                color="count",
                hover_name="country",
                hover_data={"iso3": False, "count": True},
                color_continuous_scale=[
                    [0.0, "rgba(0,120,200,0.35)"],
                    [1.0, "#00AEEF"],
                ],
                labels={"count": "Tenders"},
            )

            if not df_sel.empty:
                fig.add_trace(go.Choropleth(
                    locations=df_sel["iso3"],
                    z=[1] * len(df_sel),
                    colorscale=[[0, "#FF6B00"], [1, "#FF6B00"]],
                    showscale=False,
                    hovertext=df_sel["country"],
                    hovertemplate="%{hovertext}<extra></extra>",
                    marker_line_color="rgba(255,255,255,0.5)",
                    marker_line_width=1.0,
                ))

            fig.update_layout(
                geo=dict(
                    bgcolor="rgba(0,0,0,0)",
                    lakecolor="rgba(0,30,60,0.8)",
                    landcolor="rgba(255,255,255,0.06)",
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor="rgba(255,255,255,0.18)",
                    showland=True,
                    showcountries=True,
                    countrycolor="rgba(255,255,255,0.12)",
                    showocean=True,
                    oceancolor="rgba(0,20,50,0.7)",
                    projection=dict(type="natural earth"),
                    lataxis=dict(range=[-60, 85]),
                    lonaxis=dict(range=[-180, 180]),
                    domain=dict(x=[0, 1], y=[0, 1]),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_showscale=False,
                height=1600,
                autosize=True,
                uirevision="constant",
                dragmode=False,
                clickmode="event+select",
            )
            fig.update_traces(
                marker_line_color="rgba(255,255,255,0.25)",
                marker_line_width=0.5,
                selector=dict(type="choropleth"),
            )

            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                key="portfolio_map",
                config={"displayModeBar": False},
            )
            sel_points = (
                event.selection.get("points", [])
                if event and hasattr(event, "selection") and event.selection
                else []
            )
            clicked_iso3 = {p.get("location", "") for p in sel_points if p.get("location")}
            # Toggle: clicking the same selection again clears it
            if clicked_iso3 and clicked_iso3 == st.session_state["map_selected_iso3"]:
                st.session_state["map_selected_iso3"] = set()
            elif clicked_iso3:
                st.session_state["map_selected_iso3"] = clicked_iso3
            elif not sel_points:
                st.session_state["map_selected_iso3"] = set()

            selected_names = [
                _ISO3_TO_NAME.get(iso, iso)
                for iso in st.session_state["map_selected_iso3"]
                if _ISO3_TO_NAME.get(iso) in countries_entries
            ]

            if selected_names:
                st.markdown(
                    f'<p style="font-size:0.72rem;color:rgba(255,255,255,0.75);margin-top:0.3rem;">'
                    f'Filtro attivo: {", ".join(selected_names)}'
                    f' &nbsp;—&nbsp; <em>clicca fuori dalla selezione per resettare</em></p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="font-size:0.72rem;color:rgba(255,255,255,0.45);margin-top:0.3rem;">'
                    'Nessun paese selezionato — visualizzati tutti i tender</p>',
                    unsafe_allow_html=True,
                )
        else:
            # Fallback: no ISO3 match — plain list
            for country, entries in sorted(countries_entries.items(), key=lambda x: -len(x[1])):
                cnt = len(entries)
                st.markdown(
                    f"- {country} &nbsp;<span style='color:#7dd3fc;font-size:0.78rem;'>×{cnt}</span>",
                    unsafe_allow_html=True,
                )

    with cols[1]:
        # Filter by country selection; default = all
        if selected_names:
            filtered_ids = {id(e) for name in selected_names for e in countries_entries.get(name, [])}
            cert_c, anlzr_c, lis_c, spec_c, infra_c = _tally(
                [(e, t) for e, t in entry_tags if id(e) in filtered_ids]
            )
        else:
            cert_c, anlzr_c, lis_c, spec_c, infra_c = cert_all, anlzr_all, lis_all, spec_all, infra_all

        sections = [
            ("📋 Certificazioni", cert_c),
            ("🔬 Strumenti / Analizzatori", anlzr_c),
            ("🧬 Specialità analitiche", spec_c),
            ("💻 LIS / Protocolli IT", lis_c),
            ("🏗️ Infrastruttura & Operatività", infra_c),
        ]
        any_found = any(c for _, c in sections)

        if any_found:
            for section_label, counter in sections:
                if not counter:
                    continue
                st.markdown(
                    f'<p style="font-size:0.68rem;font-weight:700;color:rgba(255,255,255,0.55);'
                    f'margin:0.75rem 0 0.25rem;text-transform:uppercase;letter-spacing:0.07em;">'
                    f'{section_label}</p>',
                    unsafe_allow_html=True,
                )
                tags_html = ""
                for kw, cnt in counter.most_common(10):
                    cls = "insight-tag-hot" if cnt >= 2 else "insight-tag"
                    tags_html += f'<span class="{cls}">{kw} ×{cnt}</span> '
                st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:0.68rem;color:rgba(255,255,255,0.4);margin-top:0.8rem;">'
                'Tag arancioni = presenti in 2+ tender.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="font-size:0.72rem;color:rgba(255,255,255,0.45);">'
                'Nessun dato specifico trovato per la selezione.</p>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ─── LIBRARY ──────────────────────────────────────────────────────
def view_library():
    _nav("library")

    lib = load_library()

    # ── Detail view: full report for a selected entry ──────────────
    sel = st.session_state.get("lib_selected")
    if sel is not None:
        if st.button("← Back to Library", key="lib_back_detail"):
            st.session_state.lib_selected = None
            st.rerun()

        entry = lib[sel] if 0 <= sel < len(lib) else None
        if entry is None:
            st.session_state.lib_selected = None
            st.rerun()

        display_title = entry.get("display_title") or entry.get("title") or "Report"
        st.markdown(f'<div class="section-heading">{display_title}</div>', unsafe_allow_html=True)

        rpt = entry.get("report")
        if rpt:
            _render_report(rpt)
        else:
            st.info("Full report not available for this entry — it was analysed before report storage was added. Re-run the analysis to enable drill-down.")
        return

    # ── List view ──────────────────────────────────────────────────
    if st.button("← Back to Home", key="back_library"):
        st.session_state.view = "home"
        st.rerun()

    st.markdown('<div class="section-heading">Tender Library</div>', unsafe_allow_html=True)

    if not lib:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📭</div>
          <div class="empty-msg">No tenders analysed yet.<br>
          Run your first analysis to start building the library.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Portfolio Risk Insights ────────────────────────────────────
    _portfolio_insights(lib)

    # ── Stats ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analysed", len(lib))
    c2.metric("GO", sum(1 for e in lib if e.get("verdict") == "GO"))
    c3.metric("GO w/ Mitigation", sum(1 for e in lib if "Mitigation" in (e.get("verdict") or "")))
    c4.metric("NO-GO", sum(1 for e in lib if e.get("verdict") == "NO-GO"))

    st.markdown('<div class="section-heading" style="margin-top:1.8rem;">All Tenders</div>', unsafe_allow_html=True)

    # ── Full-text search (title + client + country + summary + full report JSON) ──
    search = st.text_input(
        "search",
        placeholder="🔍  Search by any word — title, client, country, ISO number, requirement…",
        label_visibility="collapsed",
    )
    filtered = lib
    if search:
        q = search.lower()
        filtered = [
            e for e in lib
            if q in (
                (e.get("title") or "") +
                (e.get("display_title") or "") +
                (e.get("client") or "") +
                (e.get("country") or "") +
                (e.get("summary") or "") +
                json.dumps(e.get("report") or {}, ensure_ascii=False)
            ).lower()
        ]

    st.caption(f"{len(filtered)} result(s)")

    lib_full = load_library()  # reload for delete operations

    for idx, entry in enumerate(filtered):
        real_idx = next(
            (i for i, e in enumerate(lib_full) if e.get("date") == entry.get("date")
             and e.get("display_title") == entry.get("display_title")),
            None,
        )
        v = entry.get("verdict", "—")
        row_cls   = "lib-row" + (" lib-row-nogo" if v == "NO-GO" else " lib-row-mit" if "Mitigation" in v else "")
        badge_cls = "badge-nogo" if v == "NO-GO" else ("badge-mit" if "Mitigation" in v else "badge-go")

        display_title = entry.get("display_title") or entry.get("client") or entry.get("title") or "—"
        date_score = f"📅 {entry.get('date','—')} &nbsp;·&nbsp; Score: {entry.get('score','—')}/100"
        has_report = bool(entry.get("report"))

        col_card, col_view, col_del = st.columns([18, 2, 1])
        with col_card:
            st.markdown(f"""
            <div class="{row_cls}">
              <div style="flex:1;min-width:0;">
                <div class="lib-client-title">{display_title}</div>
                <div class="lib-meta">{date_score}</div>
                <div class="lib-summary">{entry.get("summary","")}</div>
              </div>
              <div style="flex-shrink:0;padding-left:0.5rem;">
                <span class="lib-badge {badge_cls}">{v}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col_view:
            st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
            btn_label = "📄 Report" if has_report else "📄 —"
            btn_help  = "Open full report" if has_report else "No report stored (re-run analysis)"
            if st.button(btn_label, key=f"view_lib_{idx}", help=btn_help, disabled=not has_report):
                # Find the actual index in the full (unfiltered) library
                full_idx = next(
                    (i for i, e in enumerate(lib) if e.get("date") == entry.get("date")
                     and e.get("display_title") == entry.get("display_title")),
                    None,
                )
                if full_idx is not None:
                    st.session_state.lib_selected = full_idx
                    st.rerun()
        with col_del:
            st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
            if real_idx is not None and st.button("🗑", key=f"del_lib_{idx}", help="Delete from history"):
                lib_full.pop(real_idx)
                with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
                    json.dump(lib_full, f, ensure_ascii=False, indent=2)
                st.rerun()

    if filtered:
        st.divider()
        df_e = pd.DataFrame([
            {k: v for k, v in e.items() if k != "report"}  # skip nested JSON in CSV
            for e in filtered
        ])
        cols = [c for c in ["date", "display_title", "client", "city", "country", "verdict", "score", "summary"] if c in df_e.columns]
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

        st.markdown('<div class="kb-type-marker"></div><p style="font-size:0.8rem;color:rgba(255,255,255,0.88);font-weight:600;margin:0.6rem 0 0.1rem;">Choose what to add:</p>', unsafe_allow_html=True)
        entry_type = st.radio(
            "Choose what to add:",
            ["Showstopper", "Risk / Complexity Factor"],
            horizontal=False,
            label_visibility="collapsed",
            key="kb_entry_type",
        )
        is_ss = entry_type == "Showstopper"

        # Level selector — only for risks (showstoppers are always NO-GO, no level needed)
        risk_level = "Medium"
        if not is_ss:
            st.markdown(
                '<p style="font-size:0.8rem;color:rgba(255,255,255,0.88);font-weight:600;margin:0.7rem 0 0.2rem;">Risk level:</p>',
                unsafe_allow_html=True,
            )
            level_cols = st.columns(3)
            level_labels = {"Low": ("🟢", "#d4f5e5", "#1a7a48"),
                            "Medium": ("🟡", "#fef0c7", "#a06000"),
                            "High": ("🔴", "#fde8e8", "#c0392b")}
            if "kb_risk_level" not in st.session_state:
                st.session_state.kb_risk_level = "Medium"
            for col, lvl in zip(level_cols, ["Low", "Medium", "High"]):
                icon, bg, fg = level_labels[lvl]
                selected = st.session_state.kb_risk_level == lvl
                border = "2px solid white" if selected else "2px solid transparent"
                with col:
                    if st.button(
                        f"{icon} {lvl}",
                        key=f"kb_lvl_{lvl}",
                        use_container_width=True,
                        type="primary" if selected else "secondary",
                    ):
                        st.session_state.kb_risk_level = lvl
                        st.rerun()
            risk_level = st.session_state.kb_risk_level

        concept = st.text_area(
            "Describe the risk in plain language",
            placeholder=(
                "e.g. The tender specifies that the system must be the same brand currently installed "
                "in their lab, which is a competitor."
                if is_ss else
                "e.g. Sometimes the tender requires a connection to a specific middleware brand "
                "that we have never integrated before and the timeline is too short to develop it."
            ),
            height=100,
            key="kb_concept",
        )

        if st.button("✨ Add with AI", disabled=(not can_add or not concept.strip()), type="primary", key="kb_add_ai"):
            with st.spinner("AI is structuring the entry…"):
                try:
                    rf = load_risk_factors()
                    rf, _ = _migrate_risk_register(rf)  # ensure unified structure
                    new_entry = _ai_format_risk(
                        concept.strip(),
                        "showstopper" if is_ss else "risk_factor",
                        rf,
                        level=risk_level,
                    )
                    rr = rf.setdefault("risk_register", {})
                    if is_ss:
                        rr.setdefault("showstoppers", []).append(new_entry)
                    else:
                        rr.setdefault("risk_factors", []).append(new_entry)
                    save_risk_factors(rf)
                    level_tag = f" [{risk_level}]" if not is_ss else ""
                    st.success(f"Added **{new_entry.get('id')} — {new_entry.get('name')}**{level_tag}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        # ── Current register ─────────────────────────────────────────
        st.markdown('<div class="section-heading" style="margin-top:2rem;">Active Risk Register</div>', unsafe_allow_html=True)
        try:
            rf = load_risk_factors()
            # Migrate high_risks/medium_risks → unified risk_factors (one-time, then saved)
            rf, was_migrated = _migrate_risk_register(rf)
            if was_migrated:
                save_risk_factors(rf)
                st.toast("Risk register migrated to new format ✓", icon="✅")

            rr = rf.get("risk_register", {})
            ss_list = rr.get("showstoppers", [])
            rf_list = rr.get("risk_factors", [])

            st.caption(f"{len(ss_list)} showstoppers · {len(rf_list)} risk factors")

            with st.expander(f"🚨 Showstoppers ({len(ss_list)})"):
                if not ss_list:
                    st.markdown('<p style="color:rgba(255,255,255,0.65);font-size:0.82rem;">No showstoppers defined yet.</p>', unsafe_allow_html=True)
                for i, ss in enumerate(ss_list):
                    c_txt, c_del = st.columns([10, 1])
                    c_txt.markdown(f"**{ss.get('id','')}** — {ss.get('name','')}  \n*{ss.get('description','')}*")
                    if c_del.button("🗑", key=f"del_ss_{i}", help="Remove"):
                        rr["showstoppers"].pop(i)
                        save_risk_factors(rf)
                        st.rerun()

            lvl_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            with st.expander(f"⚠️ Risk factors ({len(rf_list)})"):
                if not rf_list:
                    st.markdown('<p style="color:rgba(255,255,255,0.65);font-size:0.82rem;">No risk factors defined yet.</p>', unsafe_allow_html=True)
                for i, r in enumerate(rf_list):
                    c_txt, c_del = st.columns([10, 1])
                    lvl = r.get("level", "")
                    lvl_tag = f" {lvl_icons.get(lvl,'')} {lvl}" if lvl else ""
                    score_tag = f" · Score: {r.get('score','?')}" if r.get("score") is not None else ""
                    c_txt.markdown(
                        f"**{r.get('id','')}** — {r.get('name','')}  {lvl_tag}{score_tag}  \n"
                        f"*{r.get('description','')}*"
                    )
                    if c_del.button("🗑", key=f"del_rf_{i}", help="Remove"):
                        rr["risk_factors"].pop(i)
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
_EMPTY_PHRASES = (
    "not specified", "not mentioned", "not provided", "not found",
    "not stated", "not indicated", "not available", "not applicable",
    "not included", "not defined", "not described", "not detailed",
    "not given", "not disclosed", "not present", "not reported",
    "not identified", "not extracted",
)

def _is_placeholder(val) -> bool:
    """Return True if val is a LLM 'nothing found' filler."""
    if not val or not isinstance(val, str):
        return False
    v = val.strip().lower()
    return any(ph in v for ph in _EMPTY_PHRASES) or v in ("n/a", "na", "none", "—", "-")


def _render_report(report: dict):
    detail = st.session_state.get("detail", "Medium")
    meta   = report.get("_meta", {})

    if meta.get("truncated"):
        st.markdown(
            f'<div class="trunc-warn">⚠️ Document truncated to fit the <b>{detail}</b> limit. '
            f'Switch to <b>High</b> for complete analysis.</div>',
            unsafe_allow_html=True,
        )

    # ── Verdict banner ─────────────────────────────────────────────
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
    def _mv(v): return "—" if _is_placeholder(v) else (v or "—")
    ttype = _mv(report.get("tender_type", ""))
    c1.metric("Tender type",  ttype.upper() if ttype != "—" else "—")
    c2.metric("Deadline",      _mv(report.get("submission_deadline", "")))
    c3.metric("Est. value",    _mv(report.get("estimated_value_eur", "")))
    city_disp    = report.get("city", "")
    country_disp = report.get("country", "")
    city_disp    = "" if _is_placeholder(city_disp) else city_disp
    country_disp = "" if _is_placeholder(country_disp) else country_disp
    loc = f"{city_disp} ({country_disp})" if city_disp and country_disp else (city_disp or country_disp or "—")
    c4.metric("Location", loc[:28])

    # ── Executive Summary ──────────────────────────────────────────
    st.markdown('<div class="section-heading">Executive Summary</div>', unsafe_allow_html=True)
    for line in report.get("executive_summary", []):
        if not _is_placeholder(line):
            st.markdown(f"- {line}")

    # ── Tender Overview (5 domain sections) ───────────────────────
    overview = report.get("tender_overview", {})
    if overview:
        st.markdown('<div class="section-heading">Tender Overview</div>', unsafe_allow_html=True)
        _OV_DOMAINS = [
            ("service_installation_support", "🔧 Service & Installation",
             "Installation scope, SLA, warranty, training, acceptance"),
            ("it_software", "💻 IT & Software",
             "LIS/HIS, middleware, protocols, cybersecurity, remote access"),
            ("commercial_legal_finance", "📑 Commercial / Legal / Finance",
             "Contract value, payment, penalties, bonds, applicable law"),
            ("layout_building_utilities", "🏗️ Layout & Building",
             "Space, floor load, utilities, civil works, compressed air"),
            ("solution_clinical_workflow", "🔬 Solution / Clinical / Workflow",
             "Automation scope, analyzers, throughput, specialties, STAT"),
        ]
        tabs = st.tabs([label for _, label, _ in _OV_DOMAINS])
        for tab, (key, label, subtitle) in zip(tabs, _OV_DOMAINS):
            with tab:
                domain = overview.get(key, {})
                if not domain:
                    st.markdown('<p class="info-box">Not extracted for this section.</p>',
                                unsafe_allow_html=True)
                    continue
                summary = domain.get("summary", "")
                if summary and not _is_placeholder(summary):
                    st.markdown(
                        f'<div style="background:rgba(0,174,239,0.08);border-left:3px solid #00AEEF;'
                        f'padding:0.7rem 1rem;border-radius:4px;margin-bottom:0.8rem;'
                        f'font-size:0.92rem;color:rgba(255,255,255,0.9);">{summary}</div>',
                        unsafe_allow_html=True,
                    )
                points = domain.get("key_points", [])
                for pt in points:
                    if not _is_placeholder(pt):
                        st.markdown(f"- {pt}")

    # ── Showstoppers ───────────────────────────────────────────────
    showstoppers = report.get("showstoppers", [])
    if showstoppers:
        st.markdown(
            f'<div class="section-heading section-heading-orange">🚨 Showstoppers ({len(showstoppers)}) — NO-GO</div>',
            unsafe_allow_html=True,
        )
        for ss in showstoppers:
            if not isinstance(ss, dict):
                st.markdown(f"- {ss}")
                continue
            doc_ref = ss.get("document_ref", "")
            ref_line = f" · 📄 {doc_ref}" if doc_ref else ""
            st.markdown(f"""
<div class="ss-card">
  <div class="ss-id">{ss.get("id", "SS")}</div>
  <div class="ss-desc">{ss.get("description", "")}</div>
  <div class="ss-evidence">Evidence: {ss.get("evidence", "—")} · Impact: {ss.get("impact", "—")}{ref_line}</div>
</div>""", unsafe_allow_html=True)

    # ── Key Deadlines & Milestones ─────────────────────────────────
    deadlines = report.get("deadlines", [])
    st.markdown('<div class="section-heading">Key Deadlines & Milestones</div>', unsafe_allow_html=True)
    if deadlines:
        for d in deadlines:
            if not isinstance(d, dict):
                if not _is_placeholder(d):
                    st.markdown(f"- {d}")
                continue
            when = d.get("when", "")
            milestone = d.get("milestone", "")
            if _is_placeholder(when) and _is_placeholder(milestone):
                continue
            ev = d.get("evidence", "")
            ev_part = f"  \n  *{ev}*" if ev and not _is_placeholder(ev) else ""
            st.markdown(f"- **{when or '?'}** — {milestone}{ev_part}")
    else:
        st.markdown('<p class="info-box">No specific dates identified in the document.</p>', unsafe_allow_html=True)

    # ── Requirements ───────────────────────────────────────────────
    reqs = report.get("requirements", {})
    st.markdown('<div class="section-heading">Requirements & Constraints</div>', unsafe_allow_html=True)
    if reqs:
        for key, items in reqs.items():
            if items:
                filtered = []
                for item in items:
                    if isinstance(item, dict):
                        txt = item.get("text", str(item))
                        if not _is_placeholder(txt):
                            filtered.append(f"- {txt}")
                    elif not _is_placeholder(item):
                        filtered.append(f"- {item}")
                if filtered:
                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                    for f_item in filtered:
                        st.markdown(f_item)
    else:
        st.markdown('<p class="info-box">No requirements extracted.</p>', unsafe_allow_html=True)

    # ── Deliverables ───────────────────────────────────────────────
    deliverables = report.get("deliverables", [])
    st.markdown('<div class="section-heading">Deliverables to Prepare</div>', unsafe_allow_html=True)
    if deliverables:
        for item in deliverables:
            if not _is_placeholder(item):
                st.markdown(f"- {item}")
    else:
        st.markdown('<p class="info-box">No deliverables listed.</p>', unsafe_allow_html=True)

    # ── Risk Register ──────────────────────────────────────────────
    risks = report.get("risks", [])
    st.markdown('<div class="section-heading">Risk Register</div>', unsafe_allow_html=True)
    if risks:
        risks_dicts = [r for r in risks if isinstance(r, dict)]
        if risks_dicts:
            df = pd.DataFrame(risks_dicts)
            ordered = [c for c in ["id", "risk", "category", "level", "score", "document_ref", "evidence", "mitigation"] if c in df.columns]
            df_sorted = df[ordered].sort_values(
                "score", ascending=False,
                key=lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
            ) if "score" in df.columns else df[ordered]
            st.dataframe(df_sorted, use_container_width=True, hide_index=True)
    else:
        st.markdown('<p class="info-box">No risks identified.</p>', unsafe_allow_html=True)

    # ── Open Questions ─────────────────────────────────────────────
    open_qs = report.get("open_questions", [])
    if open_qs:
        filtered_qs = [q for q in open_qs if not _is_placeholder(q)]
        if filtered_qs:
            st.markdown('<div class="section-heading">Open Questions</div>', unsafe_allow_html=True)
            for q in filtered_qs:
                st.markdown(f"- {q}")

    # ── Download + API usage ───────────────────────────────────────
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
                if meta.get("runs", 1) > 1:
                    st.caption(f"Runs: {meta['runs']}× consensus merge")
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
