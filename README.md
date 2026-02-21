# Inpeco · Tender Intake Assistant

> **Un commerciale riceve una gara da 8 milioni di euro. Ha 3 settimane per decidere se partecipare. Normalmente ci vogliono 2-3 giorni di riunioni, email e lettura manuale per capire se vale la pena. Con questo tool: 4 minuti.**

---

## Cos'è

Un sistema AI di **pre-screening automatico per gare d'appalto** nel settore della Total Laboratory Automation (TLA) e automazione di laboratorio clinico.

Carica il PDF della gara. In pochi minuti ricevi un report completo con:

- ✅ / ❌ **Raccomandazione Go / No-Go** motivata con punteggio
- 🚨 **Showstopper rilevati** — ragioni per cui non si dovrebbe nemmeno partecipare
- ⚠️ **Risk register pesato** — ogni rischio con probabilità, impatto e score combinato
- 📋 **Requisiti chiave estratti** — tecnici, commerciali, legali
- 📅 **Milestone e timeline** identificate nel documento
- 📝 **Report Word scaricabile** pronto per la revisione interna

---

## Perché è diverso da "chiedere a ChatGPT"

ChatGPT non sa chi è Inpeco, cosa sa fare, cosa non sa fare, e non ha memoria delle gare precedenti.

Questo sistema è **addestrato sul contesto reale di Inpeco**:

### 1. Register di rischi proprietario
Il sistema non valuta le gare in modo generico. Usa un registro costruito su misura con showstopper e fattori di rischio specifici per il business di Inpeco: spazio fisico incompatibile, richiesta di automazione blood bank (non ancora disponibile), connettività HIL via laser vision, brownfield con zero downtime, responsabilità turnkey, e molto altro.

Ogni voce ha segnali linguistici precisi — parole e frasi che, se trovate nel documento, triggherano quella regola. Il modello sa cercarle in italiano, inglese, tedesco e francese.

### 2. Impara dalle risposte passate di Inpeco
Carica nella Knowledge Base i documenti con le **risposte di Inpeco a gare precedenti** (vinte o perse). Il sistema li legge e capisce — anche dal linguaggio diplomatico — cosa Inpeco sa fare davvero e dove ha limitazioni reali.

Frasi come *"da confermare in fase di progetto"*, *"compatibile in linea di principio"*, *"soggetto a sopralluogo"* vengono riconosciute come segnali di incertezza. Il sistema trasferisce questa conoscenza istituzionale nelle analisi future.

### 3. Risk editor in linguaggio naturale
Aggiungi nuovi rischi o showstopper descrivendo il problema in italiano. L'AI lo struttura automaticamente nel formato corretto e lo aggiunge al registro attivo. Nessun JSON, nessun tecnicismo.

---

## Come funziona — il flusso

```
Gara (PDF/DOCX)
      │
      ▼
  Estrazione testo
      │
      ▼
  GPT-4o analizza contro:
  ├─ Risk register Inpeco (showstopper + risk factors)
  └─ Risposte passate di Inpeco (knowledge base)
      │
      ▼
  Report strutturato JSON
      │
      ├─ Dashboard interattiva (Streamlit)
      └─ Export Word (.docx)
```

Tre livelli di profondità analisi:
- **Low** (~2 min) — solo showstopper, go/no-go rapido
- **Medium** (~4 min) — risk register completo + requisiti
- **High** (~8 min) — analisi esaustiva, tutto il dettaglio

---

## Funzionalità principali

| Modulo | Cosa fa |
|---|---|
| **Analyse Tender** | Carica gara, inserisci API key, lancia analisi GPT-4o |
| **Tender Library** | Storico di tutte le gare analizzate, filtrabili, esportabili CSV |
| **Knowledge Base → Risk Factors** | Visualizza, aggiungi (con AI) ed elimina showstopper e risk factors |
| **Knowledge Base → Past Bid Responses** | Carica risposte Inpeco a gare passate per auto-istruire il sistema |

---

## Stack tecnico

- **Frontend**: Streamlit (Python) — interfaccia web, zero infrastruttura
- **AI**: OpenAI GPT-4o via API (chiave API propria, nessun dato inviato a terzi fuori dall'API call)
- **Estrazione documenti**: PyMuPDF (PDF), python-docx (Word), pdfplumber
- **Export**: python-docx per report Word formattato
- **Dati**: tutto locale — nessun database, nessun cloud, file JSON su disco

---

## Avvio locale

```bash
git clone <repo-url>
cd tender-intake-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Apri il browser su `http://localhost:8501`, inserisci la tua API key OpenAI e carica la prima gara.

---

## Struttura del progetto

```
tender-intake-assistant/
├── app.py                     # App Streamlit — UI e routing
├── src/
│   ├── pipeline.py            # Core: prompt GPT-4o, parsing risposta
│   ├── extractors.py          # Estrazione testo da PDF/DOCX/TXT
│   └── report_docx.py         # Generazione report Word
├── assets/
│   ├── risk_factors.json      # Register showstopper + risk factors Inpeco
│   ├── tender_library.json    # Storico gare analizzate
│   └── knowledge/
│       └── responses/         # Risposte Inpeco a gare passate
└── requirements.txt
```

---

## Il vero valore

Ogni gara a cui Inpeco risponde richiede ore di lavoro di persone qualificate per capire se vale la pena partecipare. Molte gare vengono analizzate e poi abbandonate. Alcune vengono vinte ma avevano rischi non visti in fase di pre-bid.

Questo tool non sostituisce il giudizio umano — lo potenzia. Dà al commerciale e al bid team un **punto di partenza strutturato e oggettivo in 4 minuti**, basato sulla conoscenza reale di Inpeco, non su valutazioni generiche.

Col tempo, più risposte passate vengono caricate nella knowledge base, più il sistema diventa preciso. È un loop virtuoso: ogni gara analizzata migliora la capacità di analizzare le prossime.

---

**→ [Apri l'app](https://share.streamlit.io)** *(link aggiornato al deploy)*
