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
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
)


def load_risk_factors(path: str | None = None) -> Dict[str, Any]:
    """Load company-specific risk factors.

    Uses an absolute path so it works regardless of CWD.
    If the user file does not exist yet (e.g. fresh deployment), it is
    copied from the committed seed file.
    """
    import shutil as _shutil
    if path is None:
        path = os.path.join(_ASSETS_DIR, "risk_factors.json")
    if not os.path.exists(path):
        seed = os.path.splitext(path)[0] + ".seed.json"
        if os.path.exists(seed):
            _shutil.copy(seed, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Knowledge context loader (disk fallback) ─────────────────────

def _load_knowledge_context_from_disk(
    max_chars_per_file: int = 20_000,
    max_total: int = 80_000,
) -> str:
    """Load past-bid documents from the knowledge folders and return as a single string.
    Excel compliance matrices are parsed with the dedicated parser."""
    import glob as _glob
    from .extractors import parse_bid_response_excel, extract_from_file

    chunks: list[str] = []
    total = 0
    folder_labels = [
        ("assets/knowledge/responses", "PAST BID RESPONSE"),
        ("assets/knowledge/won",  "PAST BID RESPONSE (won)"),
        ("assets/knowledge/lost", "PAST BID RESPONSE (lost)"),
    ]
    for folder, label in folder_labels:
        if not os.path.isdir(folder):
            continue
        for fpath in sorted(_glob.glob(os.path.join(folder, "*"))):
            if total >= max_total:
                break
            fn = os.path.basename(fpath)
            ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
            try:
                with open(fpath, "rb") as fh:
                    raw = fh.read()
                if ext in ("xlsx", "xls"):
                    text = parse_bid_response_excel(raw, fn)
                else:
                    pages = extract_from_file(raw, fn)
                    text = "\n".join(pages)
                text = text[:max_chars_per_file]
                chunks.append(f"=== {label}: {fn} ===\n{text}")
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

PAST BID COMPLIANCE DATA — INPECO'S HISTORICAL ANSWERS:
The following data is extracted from Inpeco's compliance matrices in past tender responses
(Excel files with requirement lists, Y/N/partially columns, mandatory/optional flags).

HOW TO READ THIS DATA:
- "MANDATORY — NOT COMPLIANT (N)": requirements Inpeco COULD NOT meet in a past bid.
  These are CONFIRMED CAPABILITY GAPS. If the current tender contains similar requirements,
  you MUST flag them as HIGH risk or SHOWSTOPPER and explicitly reference the past gap.
- "MANDATORY — PARTIALLY COMPLIANT": requirements Inpeco only partially met.
  These are KNOWN WEAKNESSES. Flag as MEDIUM-HIGH risk if similar requirements appear.
- "OPTIONAL — NOT COMPLIANT / PARTIAL": lower-priority gaps, still worth noting.
- "MANDATORY — COMPLIANT (Y)": confirmed capabilities; use to reassure where relevant.

INSTRUCTIONS:
1. Cross-reference EVERY requirement in the current tender against these past gaps.
2. When you find a match or similarity, say so explicitly in the risk evidence and rationale:
   e.g. "In past bid [filename], Inpeco answered N to a similar connectivity requirement."
3. Treat recurring N/partial patterns as systemic limitations, not one-off cases.
4. Do NOT use hedged language if the data clearly shows a gap — be direct.

{knowledge_context}

END OF PAST BID COMPLIANCE DATA
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

TENDER TYPE CLASSIFICATION — READ CAREFULLY AND APPLY STRICTLY:
- "bundle": The tender lot explicitly requires Inpeco to SUPPLY analyzers and/or reagents as part
  of the same contract. Inpeco must act as prime contractor for the full integrated solution
  including third-party IVD instruments.
  Signals: "fornitura di analizzatori", "supply of analyzers included in the lot", reagents in the
  same lot, "all-in-one contract", "automation + diagnostic instruments to be provided".

- "unbundle": The tender covers ONLY the pre-analytical automation track. The analyzers are already
  owned by the hospital OR will be procured via a completely separate tender/lot. Inpeco only
  supplies and installs the automation system and ensures connectivity to those analyzers.
  ⚠️ CRITICAL ANTI-PATTERN — do NOT classify as bundle just because:
    • The document lists analyzer brands/models (they are connectivity targets, not items to supply)
    • The document requires interfacing or connecting to existing/future analyzers
    • The document mentions middleware or LIS integration
    • The word "analizzatori" appears (it may refer to existing equipment)
  Signals: "automazione da collegare", "connectivity to existing instruments", "pre-analytics only",
  "track only", "interfacciamento con analizzatori presenti", analyzers listed only as things
  to connect to, separate lots or separate tenders for analyzers.

- "unknown": Insufficient information in the document to determine type.

RISK REGISTER TO APPLY:
{json.dumps(risk_factors.get("risk_register", {}), indent=2, ensure_ascii=False)}
{knowledge_section}
RESPONSE FORMAT:
You MUST respond with a valid JSON object. No markdown, no explanation outside JSON.
Use exactly this structure:

{{
  "tender_title": "string",
  "tender_date": "string or empty",
  "tender_reference": "string or empty",
  "contracting_authority": "string",
  "city": "string (city where contracting authority is located, or empty)",
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
    "scope_and_responsibility": ["EXHAUSTIVE list — extract EVERY explicit scope item: what is included, what is excluded, who is responsible for civil works, electrical, pneumatic, dismantling, training, go-live support. Quote the document where relevant."],
    "space_and_facility": ["EXHAUSTIVE list — extract ALL spatial constraints: room dimensions, available m², floor load, ceiling height, door widths, number of floors/buildings involved, available electrical power, compressed air availability, HVAC requirements."],
    "analyzer_connectivity": ["EXHAUSTIVE list — name EVERY analyzer/instrument mentioned with brand and model if stated. Distinguish between analyzers to be connected (unbundle) vs supplied (bundle). Note specialties: clinical chemistry, immunoassay, hematology, coagulation, urinalysis, microbiology, molecular, blood gas, etc."],
    "it_and_middleware": ["EXHAUSTIVE list — LIS/HIS name and version if stated, HL7/ASTM/FHIR requirements, cybersecurity requirements, GDPR obligations, remote access requirements, server supply obligations, network requirements."],
    "schedule_and_milestones": ["EXHAUSTIVE list — ALL dates and timelines: contract signature, site survey, delivery, installation start/end, go-live, acceptance testing, warranty period start/end. Include phased rollout if present."],
    "qualification_and_compliance": ["EXHAUSTIVE list — ALL certifications, regulatory approvals, standards required: CE marking, ISO standards (9001/13485/27001/62443), local ministry approvals, IQ/OQ/PQ validation, accreditation requirements (ISO 15189, CAP), GMP requirements."],
    "commercial_conditions": ["EXHAUSTIVE list — payment terms, penalty/liquidated damages clauses (amount and trigger), performance bonds, bank guarantees, warranty duration, SLA uptime % required, spare parts obligations, insurance requirements, exclusivity clauses."]
  }},
  "deliverables": ["EXHAUSTIVE list — list EVERY document, report, plan, certificate, training session, and formal deliverable explicitly requested in the tender. Include: technical offer, pricing schedule, compliance matrix, project plan, site survey report, FAT/SAT protocols, installation report, training plan, O&M manuals, validation documentation, as-built drawings, etc."],
  "open_questions": ["string"],
  "deadlines": [
    {{
      "milestone": "string",
      "when": "string",
      "evidence": "string"
    }}
  ],
  "tender_overview": {{
    "service_installation_support": {{
      "summary": "2-4 sentence narrative covering the overall installation and service picture: who does what, key timeline, SLA level required, training obligations.",
      "key_points": [
        "Exhaustive bullets. Cover: site survey obligations and timing; installation responsibility (Inpeco, subcontractor, hospital); dismantling/removal of existing equipment; go-live support duration; acceptance testing procedure (FAT/SAT/IQ/OQ/PQ); SLA uptime % required; response time for faults (critical vs standard); spare parts obligations; maintenance contract type (full-risk, time & material, preventive); warranty duration and start trigger; training: who is trained, how many sessions, on-site vs remote; documentation to deliver (manuals, as-built drawings, validation dossiers); post-go-live hypercare period if stated."
      ]
    }},
    "it_software": {{
      "summary": "2-4 sentence narrative on the IT integration complexity: LIS/middleware involved, protocol requirements, cybersecurity obligations, any server supply.",
      "key_points": [
        "Exhaustive bullets. Cover: LIS name and version (if stated); HIS name (if stated); middleware platform name (if stated); communication protocols required (HL7 version, ASTM, FHIR, proprietary); message types required (order, result, status, ADT); bidirectional vs unidirectional interface; cybersecurity requirements (ISO 27001, IEC 62443, NIS2, penetration testing, DPIA); GDPR/data-residency obligations; remote access requirements (VPN, jump server, whitelisting); server/hardware to supply (specs if stated); network requirements (dedicated VLAN, bandwidth, latency); software validation requirements (IQ/OQ/PQ for software modules); software certification (CE IVD, FDA 510k, MDR if applicable); update/patch management obligations; disaster recovery / backup requirements."
      ]
    }},
    "commercial_legal_finance": {{
      "summary": "2-4 sentence narrative on the commercial and contractual profile: contract value, payment structure, main financial risks (penalties, bonds), applicable law.",
      "key_points": [
        "Exhaustive bullets. Cover: estimated contract value (total and per lot if applicable); payment terms (advance, milestone-based, on delivery, on acceptance); penalty/liquidated damages clauses (trigger event, amount per day/week, cap); performance bond or bank guarantee (%, duration, issuer requirements); insurance requirements (PI, public liability, amounts); warranty duration and scope; SLA financial penalties (if uptime SLA is missed); contract duration (if service/rental model); renewal/extension options; exclusivity clauses; applicable law and jurisdiction; dispute resolution mechanism (arbitration, court, ADR); price revision/indexation clauses; import/export restrictions or custom duties if cross-border; subcontracting restrictions."
      ]
    }},
    "layout_building_utilities": {{
      "summary": "2-4 sentence narrative on the physical installation environment: available space, building constraints, utilities available, civil works scope.",
      "key_points": [
        "Exhaustive bullets. Cover: total available floor area (m²) and dimensions of the lab/room; ceiling height; floor load capacity (kg/m²); door widths and corridor widths for equipment access; number of floors or buildings involved; elevator availability and dimensions; electrical supply available (kVA, phases, voltage, UPS); compressed air availability (bar, flow rate, dedicated line or shared); HVAC / air conditioning in the lab; drainage requirements; civil works scope (who pays, who executes: false floors, cable trays, partitions); pneumatic tube system (existing, to install, brand); fire safety and regulatory constraints for the lab space; any asbestos or structural survey requirements."
      ]
    }},
    "solution_clinical_workflow": {{
      "summary": "2-4 sentence narrative on the clinical and workflow requirements: what the automation must achieve clinically, throughput, tube types, analyzers to connect or supply.",
      "key_points": [
        "Exhaustive bullets. Cover: automation solution scope (pre-analytical only, full-track, post-analytical); required throughput (tubes/hour, peak load); tube types handled (primary, secondary, aliquots, caps on/off); sample types (serum, plasma, urine, CSF, blood, other); clinical specialties to serve (clinical chemistry, immunoassay, hematology, coagulation, urinalysis, microbiology, molecular, blood gas, toxicology, genetics); STAT workflow requirements (dedicated STAT lane, priority routing); centrifugation requirements (on-track centrifuge, number, RPM, temperature); decapping/recapping requirements; aliquoting requirements (number of daughters, volume); refrigerated storage (number of positions, temperature); sorting and routing logic complexity; analyzer list: brand + model + specialty for each (specify if to connect or to supply); consolidation requirement (replacing existing instruments with new ones); interfacing to legacy analyzers (specify if connectivity is known/certified); specific clinical protocols required (e.g. reflex testing, delta-check, ASAP routing)."
      ]
    }}
  }}
}}

FIELD EXTRACTION RULES — APPLY TO EVERY ANALYSIS:
- city: Extract the city where the contracting authority/hospital is located. Look in: letterhead,
  address blocks, "sede legale", "indirizzo", jurisdiction clause, applicable law references,
  institution name (e.g. "Ospedale Civico di Palermo" → city = "Palermo"). If not explicit,
  infer from the institution name or document language. Never leave blank if inferable.
- country: Extract the country. Infer from language (Italian doc → Italy), institution name,
  applicable law ("diritto svizzero" → Switzerland, "legge italiana" → Italy), or address.
  Never leave blank if inferable.
- submission_deadline: Extract the exact date by which the bid must be submitted. Look for:
  "scadenza", "termine presentazione offerta", "data presentazione", "submission deadline",
  "offerte entro il", "deadline". Return as a readable date string (e.g. "15 March 2025").
  If not stated, return empty string.
- estimated_value_eur: Extract the estimated contract value. Look for: "valore stimato",
  "importo a base d'asta", "base d'asta", "importo complessivo", "estimated value",
  "contract value", "budget". Return as a string with currency (e.g. "€ 2.400.000").
  If not stated, return empty string.

SCORING RULES:
- risks level: Low | Medium | High based on severity/likelihood of blocking the bid
- risks score: integer 0-100 measuring risk severity
  - Low risk  → score 15–35
  - Medium risk → score 40–65
  - High risk → score 70–90
- document_ref: always specify document name (if multiple files uploaded), page number, and section/paragraph
- Go/No-Go logic:
  - Any showstopper present → NO-GO
  - Any High risk OR average risk score >= 55 → GO with Mitigation
  - Otherwise → GO
- go_nogo score: 0-100 representing overall bid feasibility (100=fully feasible, 0=completely infeasible)
- requirements and deliverables: BE EXHAUSTIVE — do not summarize or truncate. Every item explicitly mentioned in the tender must appear.

REPORT VERBOSITY — MANDATORY MINIMUM STANDARDS:
The report MUST be comprehensive and detailed. Apply these minimum standards to every field:

executive_summary:
  - Minimum 6 bullet points.
  - Each point must be a complete sentence of at least 30 words.
  - Cover: tender scope, contract value, type (bundle/unbundle), submission deadline,
    top 2-3 risks or showstoppers, knowledge-base gaps found, go/no-go recommendation rationale.

go_nogo.rationale:
  - Minimum 120 words.
  - Explain the score step by step: what drives it up, what drives it down.
  - If knowledge base gaps were found, mention them explicitly.

showstoppers (each entry):
  - description: ≥ 2 sentences explaining WHY it is a showstopper, not just WHAT it is.
  - evidence: exact quote from the document, including section reference.
  - impact: ≥ 2 sentences on business/legal/operational consequences.

risks (each entry):
  - evidence: direct quote or detailed paraphrase with section reference.
  - mitigation: concrete, actionable steps (not generic phrases like "contact the client").
    Minimum 2 sentences. Mention specific tasks, verifications, or partners if applicable.
  - If the knowledge base shows a similar gap in a past bid, say so explicitly in the evidence.

requirements sections (scope_and_responsibility, space_and_facility, etc.):
  - Do NOT group items. Each distinct sub-requirement gets its own bullet.
  - Include the relevant document section or page in parentheses where possible.
  - If a requirement is ambiguous, append "(CLARIFICATION NEEDED)" to that bullet.

open_questions:
  - List EVERY ambiguity or missing information that would affect the bid decision.
  - Each question must be specific and actionable (e.g. not "check the schedule" but
    "Confirm whether the 90-day installation deadline counts from contract signature or
    from site readiness sign-off — this distinction changes the project plan significantly.").
  - Minimum 5 open questions unless the tender is extremely complete.

deliverables:
  - Include ALL documents, certifications, plans, and reports requested.
  - Note the format required (electronic, paper, number of copies) if stated.

tender_overview — 5 DOMAIN SECTIONS (all mandatory even if some info is missing):
  Replace the single-string placeholder in each key_points array with REAL exhaustive bullets.
  Each section must be self-contained: a reader who only looks at one section must understand
  everything relevant to that domain without reading the rest of the report.

  service_installation_support:
    - Cover: site survey, installation scope (who does what), go-live support, acceptance testing
      (FAT/SAT/IQ/OQ/PQ), SLA uptime %, fault response times, spare parts, maintenance model,
      warranty duration/trigger, training (who/how many/on-site or remote), hypercare period.
    - If info is not in the tender, say explicitly: "Not specified in the tender."

  it_software:
    - Cover: LIS/HIS/middleware names and versions, protocols (HL7/ASTM/FHIR), interface direction,
      cybersecurity standards, GDPR, remote access, server supply obligations, software validation,
      certifications (CE IVD/MDR/FDA), backup/DR requirements.
    - Name every integration explicitly mentioned. If none, say so.

  commercial_legal_finance:
    - Cover: contract value, payment terms, penalties/LD (trigger + amount + cap), bonds/guarantees,
      insurance, warranty scope, SLA financial consequences, contract duration, applicable law,
      dispute resolution, exclusivity, subcontracting limits.
    - Extract exact amounts and percentages where stated.

  layout_building_utilities:
    - Cover: m², room dimensions, ceiling height, floor load, door/corridor widths, floors involved,
      electrical supply (kVA/phases/UPS), compressed air, HVAC, drainage, civil works scope,
      pneumatic tube system, structural/asbestos surveys.
    - If info is not in the tender, say explicitly: "Not specified in the tender."

  solution_clinical_workflow:
    - Cover: automation scope (pre/full/post-analytical), throughput (tubes/hour), tube types,
      sample types, clinical specialties, STAT workflow, centrifugation, decapping, aliquoting,
      refrigerated storage, analyzer list (brand + model + specialty + connect/supply), legacy
      interfacing, consolidation, specific clinical protocols (reflex, delta-check, ASAP).
    - List EVERY analyzer mentioned with its role (to connect vs to supply).
"""


def _build_user_prompt(document_text: str, detail: str = "Medium") -> str:
    detail_instructions = {
        "Low": (
            "Provide a focused analysis covering showstoppers, the top 5 risks, "
            "and a concise executive summary. Apply the verbosity standards from the system prompt "
            "even at this level — keep it concise but never vague."
        ),
        "Medium": (
            "Provide a thorough analysis. Cover ALL risk categories, ALL requirement sections, "
            "and ALL open questions. Apply the full verbosity standards from the system prompt. "
            "The report should be detailed enough that a bid manager can make a go/no-go decision "
            "without reading the original tender document."
        ),
        "High": (
            "Provide the most exhaustive analysis possible. Extract every constraint, requirement, "
            "risk, deadline, and open question. Leave nothing implicit. "
            "Apply all verbosity standards from the system prompt to their maximum extent. "
            "If the knowledge base contains past gaps relevant to ANY requirement in this tender, "
            "reference them explicitly. The report will be used as the primary briefing document "
            "for the bid team — completeness is more important than brevity."
        ),
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
