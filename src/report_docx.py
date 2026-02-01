import io
from __future__ import annotations
import datetime
from typing import Dict, Any, List, Tuple
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def _hex_to_rgb(hexstr: str) -> Tuple[int,int,int]:
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i+2], 16) for i in (0,2,4))

def _set_cell_shading(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill_hex.lstrip('#'))
    tcPr.append(shd)

def _set_cell_text(cell, text: str, bold: bool=False, color: str|None=None, size_pt: int=9):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = RGBColor(*_hex_to_rgb(color))

def _add_colored_heading(doc: Document, text: str, level: int, color_hex: str):
    p = doc.add_paragraph()
    style_name = f"CustomHeading{level}"
    existing = [s.name for s in doc.styles]
    if style_name not in existing:
        st = doc.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = doc.styles[f"Heading {min(level,9)}"]
        st.font.color.rgb = RGBColor(*_hex_to_rgb(color_hex))
        st.font.bold = True
    p.style = doc.styles[style_name]
    run = p.add_run(text)
    run.font.color.rgb = RGBColor(*_hex_to_rgb(color_hex))
    run.bold = True
    return p

def build_docx(report: Dict[str, Any], primary_hex: str, accent_hex: str) -> bytes:
    doc = Document()

    title = doc.add_paragraph()
    r = title.add_run("Tender Intake Report (Pre-Bid)")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor(*_hex_to_rgb(primary_hex))
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    subtitle = doc.add_paragraph()
    r2 = subtitle.add_run(f"{report.get('tender_title','')} — {report.get('tender_date','')}")
    r2.italic = True
    r2.font.size = Pt(11)

    doc.add_paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    _add_colored_heading(doc, "1. Executive Summary", 1, primary_hex)
    for line in report.get("executive_summary", []):
        doc.add_paragraph(line, style="List Bullet")

    _add_colored_heading(doc, "2. Key Deadlines & Milestones", 1, primary_hex)
    deadlines = report.get("deadlines", [])
    if deadlines:
        table = doc.add_table(rows=1, cols=3)
        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        hdr = table.rows[0].cells
        for i,h in enumerate(["Milestone", "When", "Evidence"]):
            _set_cell_text(hdr[i], h, bold=True, color="ffffff")
            _set_cell_shading(hdr[i], primary_hex)
        for d in deadlines:
            row = table.add_row().cells
            _set_cell_text(row[0], d.get("milestone",""))
            _set_cell_text(row[1], d.get("when",""))
            _set_cell_text(row[2], d.get("evidence",""))
    else:
        doc.add_paragraph("No explicit deadlines detected.", style="List Bullet")

    _add_colored_heading(doc, "3. Requirements & Constraints (Extracted)", 1, primary_hex)
    reqs = report.get("requirements", {})
    for cat, items in reqs.items():
        _add_colored_heading(doc, cat, 2, accent_hex)
        for it in items:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(it["text"])
            if it.get("evidence"):
                p.add_run(f" ({it['evidence']})").italic = True

    _add_colored_heading(doc, "4. Deliverables to Prepare (Pre-Bid)", 1, primary_hex)
    for item in report.get("deliverables", []):
        doc.add_paragraph(item, style="List Bullet")

    _add_colored_heading(doc, "5. Risk Register (Top)", 1, primary_hex)
    risks = report.get("risks", [])
    table = doc.add_table(rows=1, cols=7)
    hdr = table.rows[0].cells
    headers = ["ID","Risk","Category","Prob (1-5)","Impact (1-5)","Score","Evidence"]
    for i,h in enumerate(headers):
        _set_cell_text(hdr[i], h, bold=True, color="ffffff")
        _set_cell_shading(hdr[i], primary_hex)
    for rsk in risks:
        row = table.add_row().cells
        _set_cell_text(row[0], str(rsk.get("id","")))
        _set_cell_text(row[1], str(rsk.get("risk","")))
        _set_cell_text(row[2], str(rsk.get("category","")))
        _set_cell_text(row[3], str(rsk.get("prob","")))
        _set_cell_text(row[4], str(rsk.get("impact","")))
        _set_cell_text(row[5], str(rsk.get("score","")))
        _set_cell_text(row[6], str(rsk.get("evidence","")))

    _add_colored_heading(doc, "6. Go / No-Go Recommendation", 1, primary_hex)
    gn = report.get("go_nogo", {})
    doc.add_paragraph(f"Overall complexity score: {gn.get('score','?')} / 100")
    rec_p = doc.add_paragraph()
    rec_run = rec_p.add_run(f"Recommendation: {gn.get('recommendation','')}")
    rec_run.bold = True
    rec_run.font.size = Pt(14)
    rec_run.font.color.rgb = RGBColor(*_hex_to_rgb(accent_hex if gn.get('recommendation')!='GO' else primary_hex))
    if gn.get("rationale"):
        doc.add_paragraph(str(gn["rationale"]))

    buf = io.BytesIO()
doc.save(buf)
buf.seek(0)
return buf.getvalue()
