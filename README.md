# Tender Intake Assistant

AI-powered pre-bid screening for Total Laboratory Automation tenders. Upload a tender document, get a structured Go / No-Go recommendation with risk register, key requirements and a downloadable Word report — in minutes, not days.

---

## The problem

Evaluating whether to respond to a tender requires hours of careful reading, internal alignment and expert judgment. Many tenders are analysed and then dropped. Some are won with risks that were never properly flagged at the pre-bid stage.

This tool compresses that initial evaluation from days to minutes — without sacrificing the quality of the assessment.

---

## How it works

The tender document is extracted and passed to GPT-4o alongside two sources of proprietary context:

**Company-specific risk register**
A curated register of evaluation criteria — showstoppers and risk factors — built around Inpeco's specific business constraints and product capabilities. Each entry includes precise linguistic signals in multiple languages that the model actively searches for in the document.

**Institutional memory from past bids**
Upload Inpeco's written responses to previous tenders. The system reads them to learn what the company can confidently commit to, and to recognise the diplomatic language that often signals real limitations. This knowledge carries forward into every future analysis.

The result is an assessment that is not generic: it reflects the company's actual context, history and constraints.

---

## Output

Each analysis produces:

- **Go / No-Go / Go with mitigation** recommendation with score and rationale
- Showstoppers identified with textual evidence from the document
- Weighted risk register with probability, impact and combined score per factor
- Key technical, commercial and legal requirements extracted
- Milestones and deadlines identified
- Formatted Word report, ready to download

Three depth levels: **Low** (~2 min), **Medium** (~4 min), **High** (~8 min).

---

## Modules

| | |
|---|---|
| **Analyse Tender** | Upload document, run GPT-4o analysis, view interactive report |
| **Tender Library** | Full history of analysed tenders, searchable, exportable as CSV |
| **Risk Factors & Showstoppers** | Manage the evaluation register — add entries in plain language, no JSON |
| **Past Bid Responses** | Upload past written responses to enrich the knowledge base |

---

## Stack

- Python · Streamlit
- OpenAI GPT-4o — bring your own API key, no data shared with third parties
- PyMuPDF · pdfplumber · python-docx
- Fully local: no database, no cloud dependency, files on disk

---

## Setup

```bash
git clone https://github.com/lisaraco94-hub/tender-intake-assistant
cd tender-intake-assistant

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open `http://localhost:8501`, enter your OpenAI API key and upload the first tender.

---

## Design note

The system improves over time. Each set of past bid responses added to the knowledge base refines the model's ability to distinguish confident commitments from hedged ones — turning institutional experience into a structured analytical asset.

---

[github.com/lisaraco94-hub/tender-intake-assistant](https://github.com/lisaraco94-hub/tender-intake-assistant)
