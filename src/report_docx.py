import io
import datetime
from typing import Dict, Any, Tuple

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _hex_to_rgb(hexstr: str) -> Tuple[int, int, int]:
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))


def _set_cell_shading(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill_hex.lstrip("#"))
    tcPr.append(shd)


def _set_cell_text(cell, text: str, bold: bool = False, color: str | None = None, size_pt: int = 9):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = RGBColor(*_hex_to_rgb(color))


def build_docx(report: Dict[str, Any], primary_hex: str, accent_hex: str) -> bytes:
    doc = Document()

    # Title
    title = doc.add_paragraph()
    r = title.add_run("Tender Intake Report (Pre-Bid)")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor(*_hex_to_rgb(primary_hex))
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Summary
    doc.add_paragraph()
    for line in report.get("executive_summary", []):
        doc.add_paragraph(line, style="List Bullet")

    # Risks
    doc.add_paragraph("\nTop Risks")
    risks = report.get("risks", [])
    if risks:
        table = doc.add_table(rows=1, cols=4)
        hdr = table.rows[0].cells
        for i, h in enumerate(["ID", "Risk", "Prob", "Impact"]):
            _set_cell_text(hdr[i], h, bold=True, color="ffffff")
            _set_cell_shading(hdr[i], primary_hex)

        for rsk in risks:
            row = table.add_row().cells
            _set_cell_text(row[0], rsk.get("id", ""))
            _set_cell_text(row[1], rsk.get("risk", ""))
            _set_cell_text(row[2], str(rsk.get("prob", "")))
            _set_cell_text(row[3], str(rsk.get("impact", "")))
    else:
        doc.add_paragraph("No major risks detected.")

    # Save in memory
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
