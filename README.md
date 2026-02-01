# Tender Intake Assistant (Pre-Bid)

A lightweight **pre-bid screening tool** for lab automation tenders:
upload a PDF → extract key constraints, deliverables, a rule-based risk register, and a Go/No-Go suggestion → download a **Word (.docx)** report.

## Why this exists
Bid/solution teams often need a **fast first-pass** to spot show-stoppers:
- brownfield replacement during ongoing operations
- space / building constraints
- mandatory integrations (e.g., biobank connection, multi-brand analyzers)
- timeline pressure and deliverable checklist

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- No customer data is included; you upload your own tender PDF.
- Brand colors are derived from an *Inpeco-inspired* datasheet palette (see `assets/brand_style.json`).

## Roadmap ideas
- Add optional LLM summarization (guarded by API key) for better executive summary / smarter risk wording
- Multi-file intake (PDF + annexes)
- Export additional CSVs (risk register, requirements table)

