
import json
import streamlit as st
from src.pipeline import read_pdf_pages, build_prebid_report
from src.report_docx import build_docx

st.set_page_config(page_title="Tender Intake Assistant (Pre-Bid)", layout="wide")

st.title("Tender Intake Assistant (Pre-Bid)")
st.caption("Upload a tender PDF → extract constraints, deliverables, risk register, and Go/No-Go suggestion → download Word report.")

with open("assets/brand_style.json","r", encoding="utf-8") as f:
    brand = json.load(f)

col1, col2 = st.columns([1,1])
with col1:
    st.markdown("### Input")
    pdf_file = st.file_uploader("Upload tender PDF", type=["pdf"])
    detail = st.select_slider("Detail level", options=["Low","Medium","High"], value="Medium")
with col2:
    st.markdown("### Output")
    st.markdown(f"Brand style: **{brand['brand_name']}**")
    st.color_picker("Primary color", brand["primary_hex"], disabled=True)
    st.color_picker("Accent color", brand["accent_hex"], disabled=True)

if pdf_file and st.button("Run Intake"):
    pdf_bytes = pdf_file.read()
    pages = read_pdf_pages(pdf_bytes)
    report = build_prebid_report(pages)

    st.success("Report generated.")
    st.markdown("### Preview")
    st.write(report["executive_summary"])
    st.markdown("**Top risks:**")
    st.dataframe(report["risks"], use_container_width=True)

    out_name = "Tender_Intake_Report.docx"
docx_bytes = build_docx(report, brand["primary_hex"], brand["accent_hex"])

st.download_button(
    "Download Word report",
    data=docx_bytes,
    file_name=out_name,
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)