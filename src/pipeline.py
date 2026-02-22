"""
pipeline.py — AI-powered tender analysis for TLA/IVD tenders.
Replaces all rule-based heuristics with structured GPT-4o analysis
driven by a configurable risk_factors.json file.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import fitz  # PyMuPDF
from openai import OpenAI, RateLimitError

from .extractors import chunk_pages, extract_raw_text, guess_title_and_date

# ─── OpenAI client (lazy init) ────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        _client = OpenAI(api_key=api_key)
    return _client


# ─── PDF reading ──────────────────────────────────────────────────

def read_pdf_pages(pdf_bytes: bytes) -> List[str]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [doc[i].get_text("text") for i in range(len(doc))]


# ─── Risk factors loader ──────────────────────────────────────────

def load_risk_factors(path: str = "assets/risk_factors.json") -> Dict[str, Any]:
    """Load company-specific risk factors from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Knowledge context loader (disk fallback) ─────────────────────

def _load_knowledge_context_from_disk(
    max_chars_per_file: int = 12_000,
    max_total: int = 36_000,
) -> str:
    """Load past-bid documents from the knowledge folders and return as a single string."""
    import glob as _glob
    chunks: list[str] = []
    total = 0
    for folder in ("assets/knowledge/won", "assets/knowledge/lost"):
        if not os.path.isdir(folder):
            continue
        label = "WON" if folder.endswith("won") else "LOST"
        for fpath in sorted(_glob.glob(os.path.join(folder, "*"))):
            if total >= max_total:
                break
            try:
                with open(fpath, "rb") as fh:
                    raw = fh.read()
                text = raw.decode("utf-8", errors="replace")[:max_chars_per_file]
                chunks.append(f"--- [{label}] {os.path.basename(fpath)} ---\n{text}")
                total += len(text)
            except Exception:
                continue
    return "\n\n".join(chunks)


# ─── AI prompt builder ────────────────────────────────────────────

def _build_system_prompt(risk_factors: Dict[str, Any], knowledge_context: str = "") -> str:
    company = risk_factors.get("company_profile", {})

    knowledge_section = ""
    if knowledge_context.strip():
        knowledge_section = f"""

PAST BID EXPERIENCE — INPECO'S OWN RESPONSES:
The following excerpts are from Inpeco's actual written responses to past tenders (won and lost).
Read them carefully and use them to:
1. Understand Inpeco's REAL capabilities vs theoretical ones — what was actually committed vs hedged
2. Spot soft/diplomatic language that reveals true limitations:
   e.g. "subject to site survey", "to be evaluated case by case", "in principle compatible",
   "we propose to assess during project kick-off", "subject to final confirmation"
3. Flag if this new tender requires capabilities that Inpeco struggled with in past responses
4. Reference these institutional patterns when rating risks and writing the rationale

{knowledge_context}

END OF PAST BID EXPERIENCE
"""

    return f"""You are an expert pre-bid tender analyst for {company.get("name", "the company")}.

COMPANY PROFILE:
- Business: {company.get("business_description", "Clinical laboratory automation supplier")}
- Products: {", ".join(company.get("products", []))}
- Typical delivery time: {company.get("typical_delivery_months", "N/A")} months
- Geographic coverage: {", ".join(company.get("geographic_coverage", []))}
- Languages handled: {", ".join(company.get("languages", ["English"]))}

YOUR TASK:
Analyze the tender document provided and produce a structured pre-bid screening report.
Focus on identifying SHOWSTOPPERS (reasons to immediately decline), HIGH RISKS, and MEDIUM RISKS
based on the company risk register provided.

RISK REGISTER TO APPLY:
{json.dumps(risk_factors.get("risk_register", {}), indent=2, ensure_ascii=False)}

TENDER TYPE CONTEXT:
{json.dumps(risk_factors.get("tender_type_guidance", {}), indent=2, ensure_ascii=False)}
{knowledge_section}
RESPONSE FORMAT:
You MUST respond with a valid JSON object. No markdown, no explanation outside JSON.
Use exactly this structure:

{{
  "tender_title": "string",
  "tender_date": "string or empty",
  "tender_reference": "string or empty",
  "contracting_authority": "string",
  "city": "string (city where contracting authority is located, search context clues, or empty)",
  "country": "string (country, e.g. Italy, France, or empty)",
  "tender_type": "bundle | unbundle | unknown",
  "estimated_value_eur": "string or empty",
  "submission_deadline": "string or empty",
  "executive_summary": ["string", "string", "string"],
  "go_nogo": {{
    "recommendation": "GO | GO with Mitigation | NO-GO",
    "score": 0,
    "rationale": "string"
  }},
  "showstoppers": [
    {{
      "id": "string",
      "description": "string",
      "evidence": "string (direct quote or paraphrase)",
      "document_ref": "string (e.g. 'Technical Specs, p.12, §3.4')",
      "impact": "string"
    }}
  ],
  "risks": [
    {{
      "id": "string",
      "risk": "string",
      "category": "string",
      "level": "Low | Medium | High",
      "score": 50,
      "document_ref": "string (e.g. 'Tender Doc, p.8, §2.3 – Connectivity')",
      "evidence": "string (direct quote or paraphrase)",
      "mitigation": "string"
    }}
  ],
  "requirements": {{
    "scope_and_responsibility": ["string"],
    "space_and_facility": ["string"],
    "analyzer_connectivity": ["string"],
    "it_and_middleware": ["string"],
    "schedule_and_milestones": ["string"],
    "qualification_and_compliance": ["string"],
    "commercial_conditions": ["string"]
  }},
  "deliverables": ["string"],
  "open_questions": ["string"],
  "deadlines": [
    {{
      "milestone": "string",
      "when": "string",
      "evidence": "string"
    }}
  ]
}}

SCORING RULES:
- risks level: Low | Medium | High based on severity/likelihood of blocking the bid
- risks score: integer 0-100 measuring risk severity
  - Low risk  → score 15–35
  - Medium risk → score 40–65
  - High risk → score 70–90
- document_ref: always specify document name (if multiple files), page number, and section/paragraph
- Go/No-Go logic:
  - Any showstopper present → NO-GO
  - Any High risk OR average risk score >= 55 → GO with Mitigation
  - Otherwise → GO
- go_nogo score: 0-100 representing overall bid feasibility (100=fully feasible, 0=completely infeasible)
- city and country: extract from document context (letterhead, address, jurisdiction clauses); if not explicit, infer from language, law references, or institution names
"""


def _build_user_prompt(document_text: str, detail: str = "Medium") -> str:
    detail_instructions = {
        "Low": "Provide a concise analysis. Focus only on showstoppers and top 3 risks.",
        "Medium": "Provide a balanced analysis. Cover all risk categories and key requirements.",
        "High": "Provide an exhaustive analysis. Extract every constraint, requirement, and risk. "
                "List all open questions that need clarification before bidding.",
    }
    instruction = detail_instructions.get(detail, detail_instructions["Medium"])

    return f"""TENDER DOCUMENT:
{document_text}

---
ANALYSIS DETAIL LEVEL: {detail}
{instruction}

Analyze this tender document according to your system instructions and return the JSON report.
"""


# ─── Main analysis function ───────────────────────────────────────

def build_prebid_report(
    pages: List[str],
    risk_factors: Dict[str, Any] | None = None,
    detail: str = "Medium",
    knowledge_context: str = "",
) -> Dict[str, Any]:
    """
    Main entry point. Analyzes a tender document using GPT-4o and returns
    a structured report dict.

    Args:
        pages: List of page texts extracted from the PDF.
        risk_factors: Company risk register dict. If None, loads from default path.
        detail: "Low" | "Medium" | "High"

    Returns:
        Structured report dict.
    """
    if risk_factors is None:
        risk_factors = load_risk_factors()

    # Auto-load knowledge context from disk if not provided by caller
    if not knowledge_context:
        try:
            knowledge_context = _load_knowledge_context_from_disk()
        except Exception:
            pass

    client = _get_client()

    # Build full text — truncation limit depends on detail level:
    # Low  → 80k chars  (~20k tokens)  — fast & cheap, good for bulk screening
    # Medium → 200k chars (~50k tokens) — balanced
    # High → 400k chars (~100k tokens) — exhaustive, fits GPT-4o 128k context
    MAX_TEXT = {"Low": 80_000, "Medium": 200_000, "High": 400_000}.get(detail, 200_000)

    full_text = extract_raw_text(pages)
    fallback_title, fallback_date = guess_title_and_date(pages)

    truncated = len(full_text) > MAX_TEXT
    if truncated:
        full_text = full_text[:MAX_TEXT] + "\n\n[Document truncated — increase detail level for full analysis]"

    system_prompt = _build_system_prompt(risk_factors, knowledge_context)
    user_prompt = _build_user_prompt(full_text, detail)

    # Retry up to 3 times, halving the document text on each 429 "too large" error.
    for _attempt in range(4):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            break
        except RateLimitError as exc:
            if "tokens" not in str(exc) or _attempt == 3:
                raise
            # Request too large — halve the document text and rebuild prompt
            full_text = full_text[: len(full_text) // 2] + "\n\n[Document truncated due to API token limits — upgrade OpenAI plan for longer analysis]"
            user_prompt = _build_user_prompt(full_text, detail)

    raw = response.choices[0].message.content
    report = json.loads(raw)

    # Ensure fallback metadata
    if not report.get("tender_title") or report["tender_title"] == "string":
        report["tender_title"] = fallback_title
    if not report.get("tender_date"):
        report["tender_date"] = fallback_date

    # Add token usage info
    report["_meta"] = {
        "model": "gpt-4o",
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "estimated_cost_usd": round(
            (response.usage.prompt_tokens * 2.50 / 1_000_000)
            + (response.usage.completion_tokens * 10.00 / 1_000_000),
            4,
        ),
        "detail_level": detail,
        "pages_analyzed": len(pages),
        "chars_analyzed": min(len(extract_raw_text(pages)), MAX_TEXT),
        "truncated": truncated,
    }

    return report
