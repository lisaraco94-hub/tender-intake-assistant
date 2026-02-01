\
from __future__ import annotations
from typing import Dict, Any, List
import fitz  # PyMuPDF
from .extractors import extract_bullets_multiline, detect_milestones, EvidenceItem, simple_risk_register

def read_pdf_pages(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [doc[i].get_text("text") for i in range(len(doc))]

def build_prebid_report(pages: List[str]) -> Dict[str, Any]:
    """
    MVP logic:
    - Extract bullets and key phrases
    - Categorize into requirement buckets
    - Extract milestones
    - Compute rule-based risks + go/no-go
    """
    # Heuristic: first page title line(s)
    title = "Tender Document"
    date = ""
    if pages:
        lines = [l.strip() for l in pages[0].splitlines() if l.strip()]
        if lines:
            title = lines[0][:120]
        # find date-like token
        for l in lines[:10]:
            if any(ch.isdigit() for ch in l) and "." in l:
                date = l
                break

    req: Dict[str, List[EvidenceItem]] = {
        "Project scope & responsibility": [],
        "Required content in project description": [],
        "Operational / brownfield constraints": [],
        "Facility / space constraints": [],
        "Analyzer connectivity & layout rules": [],
        "Schedule / milestones": [],
        "Provided drawings / supplements": [],
    }

    milestones = []
    deliverables = []

    for idx, page_text in enumerate(pages, start=1):
        bullets = extract_bullets_multiline(page_text)
        lower = page_text.lower()

        # Milestones
        for m in detect_milestones(page_text):
            milestones.append({"milestone": m["milestone"], "when": m["when"], "evidence": f"p.{idx}"})

        # Categorization heuristics
        for b in bullets:
            ev = f"p.{idx}"
            lb = b.lower()
            item = EvidenceItem(text=b, evidence=ev)

            if "layout" in lb or "parts list" in lb or "project scheduling" in lb:
                req["Required content in project description"].append(item)
            if "turnkey" in lower or "third-party" in lower:
                # add once per doc if found
                if not req["Project scope & responsibility"]:
                    req["Project scope & responsibility"].append(EvidenceItem(
                        text="Bidder responsible for entire system; delivers a turnkey system; third-party providers contracted directly by bidder.",
                        evidence=ev
                    ))

            if "ongoing operations" in lb or "commissioning" in lb or "sample collection area" in lb:
                req["Operational / brownfield constraints"].append(item)

            if "space" in lb or "area" in lb or "door" in lb or "ceiling" in lb or "floor" in lb or "biobank" in lb or "connection" in lb:
                req["Facility / space constraints"].append(item)

            if "analytics" in lb or "connected" in lb or "validation workstation" in lb or "reserve" in lb or "cobas" in lb or "sysmex" in lb:
                req["Analyzer connectivity & layout rules"].append(item)

            if "dwg" in lb or "file name" in lb or "floor plan" in lb or "master plan" in lb or "appendix" in lb:
                req["Provided drawings / supplements"].append(item)

            if "signing" in lb or "installation" in lb or "training" in lb or "middleware" in lb:
                req["Schedule / milestones"].append(item)

    # Deliverables (baseline)
    deliverables = [
        "High-level feasibility assessment and assumptions log (highlight deviations vs tender framework).",
        "High-level layout concept(s) addressing all space constraints and required analyzer connections.",
        "Project schedule (planning → delivery → installation → testing & commissioning).",
        "Parts list / BOM of offered components (high level).",
        "Responsibility & interface matrix for any third parties (turnkey scope).",
        "Drawing / data request list (DWG, floor plans, biobank interface points, etc.).",
    ]

    # Risks
    risks = simple_risk_register(req)

    # Go/no-go score from top risks
    max_score = 25 * max(1, len(risks))
    raw = sum(r["score"] for r in risks)
    complexity = int(round((raw / max_score) * 100)) if max_score else 0
    if complexity >= 70:
        recommendation = "NO-GO"
    elif complexity >= 45:
        recommendation = "GO with Mitigation"
    else:
        recommendation = "GO"

    executive_summary = [
        f"Document parsed: {title}.",
        "Output intended for pre-bid screening: main constraints, show-stoppers, required deliverables, and a rule-based risk register.",
        f"Preliminary recommendation: {recommendation} (complexity {complexity}/100).",
    ]

    go_nogo = {
        "score": complexity,
        "recommendation": recommendation,
        "rationale": "Recommendation is driven by risk register scores (rule-based). Validate with early feasibility checks and site/drawing confirmation."
    }

    # Convert EvidenceItem lists to dicts
    req_out: Dict[str, List[Dict[str,str]]] = {}
    for k, items in req.items():
        # de-dup
        seen=set()
        out=[]
        for it in items:
            key=(it.text, it.evidence)
            if key in seen: 
                continue
            seen.add(key)
            out.append({"text": it.text, "evidence": it.evidence})
        if out:
            req_out[k]=out

    # Deduplicate milestones
    ms_seen=set()
    ms_out=[]
    for m in milestones:
        key=(m["milestone"], m["when"])
        if key in ms_seen:
            continue
        ms_seen.add(key)
        ms_out.append(m)

    return {
        "tender_title": title,
        "tender_date": date,
        "executive_summary": executive_summary,
        "deadlines": ms_out,
        "requirements": req_out,
        "deliverables": deliverables,
        "risks": risks,
        "go_nogo": go_nogo,
    }
