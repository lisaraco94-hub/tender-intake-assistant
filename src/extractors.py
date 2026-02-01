\
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class EvidenceItem:
    text: str
    evidence: str  # e.g. "p.3"

BULLET_PAT = re.compile(r"^\s*•\s*(.*)\s*$")

def extract_bullets_multiline(text: str) -> List[str]:
    """
    Extract bullet points from text where the bullet marker '•' may be followed by wrapped lines.
    """
    lines = [l.rstrip() for l in text.splitlines()]
    bullets: List[str] = []
    cur: Optional[str] = None
    in_bullet = False

    for line in lines:
        m = BULLET_PAT.match(line.strip())
        if m:
            if cur:
                bullets.append(cur.strip())
            cur = (m.group(1) or "").strip()
            in_bullet = True
            continue

        if in_bullet:
            if line.strip() == "":
                continue
            cur = (cur or "").strip()
            cur = (cur + " " + line.strip()).strip()

    if cur:
        bullets.append(cur.strip())

    return [b for b in bullets if b]

def detect_milestones(text: str) -> List[Dict[str, str]]:
    """
    Lightweight milestone extractor: looks for quarters and month-year phrases.
    """
    milestones = []
    # Quarter like 2Q2025 / Q3 2025 / 3Q2025
    q_pat = re.compile(r"(?P<q>[1-4])\s*Q\s*(?P<y>20\d{2})|(?P<q2>[1-4])Q(?P<y2>20\d{2})", re.IGNORECASE)
    # Month names (English/German) + year
    month_pat = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b\s*(20\d{2})", re.IGNORECASE)

    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        if "contract" in l.lower() or "installation" in l.lower() or "training" in l.lower() or "commission" in l.lower() or "signing" in l.lower():
            mq = q_pat.search(l)
            mm = month_pat.search(l)
            when = None
            if mq:
                q = mq.group("q") or mq.group("q2")
                y = mq.group("y") or mq.group("y2")
                when = f"{q}Q{y}"
            elif mm:
                when = f"{mm.group(1)} {mm.group(2)}"
            if when:
                milestones.append({"milestone": l, "when": when})
    return milestones

def simple_risk_register(requirements_by_cat: Dict[str, List[EvidenceItem]]) -> List[Dict[str, Any]]:
    """
    Rule-based risk register. You can expand this list over time.
    """
    risks = []
    def add(rid, risk, cat, p, i, ev):
        risks.append({"id": rid, "risk": risk, "category": cat, "prob": p, "impact": i, "score": p*i, "evidence": ev})

    # Brownfield / ongoing ops
    for it in requirements_by_cat.get("Operational / brownfield constraints", []):
        if "ongoing" in it.text.lower() or "ongoing operations" in it.text.lower():
            add("R1", "Replacement during ongoing operations (brownfield). Cutover/phasing complexity and continuity risk.",
                "Operational / Timeline", 4, 5, it.evidence)
            break

    # Turnkey / third parties
    for it in requirements_by_cat.get("Project scope & responsibility", []):
        if "turnkey" in it.text.lower() or "third-party" in it.text.lower():
            add("R2", "Turnkey responsibility incl. third-party contracting: commercial/legal exposure and integration risk.",
                "Commercial / Legal", 3, 4, it.evidence)
            break

    # Space / facility constraints
    for it in requirements_by_cat.get("Facility / space constraints", []):
        if "space" in it.text.lower() or "area" in it.text.lower() or "ceiling" in it.text.lower() or "door" in it.text.lower():
            add("R3", "Tight facility constraints (space/door/ceiling). Potential show-stopper for layout and installation logistics.",
                "Facility / Layout", 4, 4, it.evidence)
            break

    # Multi-floor / biobank connection
    for it in requirements_by_cat.get("Facility / space constraints", []):
        if "biobank" in it.text.lower() or "vertical" in it.text.lower() or "connection" in it.text.lower():
            add("R4", "Mandatory biobank connection / multi-floor integration increases technical and project risk.",
                "Technical / Facility", 4, 4, it.evidence)
            break

    # Analyzer integration complexity
    for it in requirements_by_cat.get("Analyzer connectivity & layout rules", []):
        if "connected" in it.text.lower() or "validation workstation" in it.text.lower() or "reserve" in it.text.lower():
            add("R5", "High analyzer integration scope + validation workstations may inflate footprint and workflow complexity.",
                "Technical / Workflow", 3, 4, it.evidence)
            break

    # Keep top by score
    risks.sort(key=lambda r: r["score"], reverse=True)
    return risks[:10]
