# Tender Intake Assistant

Pre-bid screening automatico per gare d'appalto nel settore della Total Laboratory Automation. Carica il documento di gara, ottieni in pochi minuti un'analisi strutturata con raccomandazione Go / No-Go, risk register pesato, requisiti chiave e report Word scaricabile.

---

## Il problema

Valutare se partecipare a una gara richiede ore di lettura, confronto interno e giudizio esperto. Molte gare vengono analizzate e poi abbandonate. Alcune vengono vinte con rischi che non erano stati visti in fase di pre-bid.

Questo tool comprime quella prima valutazione da giorni a minuti, senza sacrificare la qualità del giudizio.

---

## Come funziona

Il documento di gara viene estratto e passato a GPT-4o insieme a due fonti di conoscenza proprietaria:

**Risk register Inpeco**
Un registro costruito su misura con showstopper e fattori di rischio specifici per il business: spazio fisico incompatibile, richiesta di automazione blood bank, connettività HIL via laser vision, installazioni brownfield con zero downtime, responsabilità turnkey, scadenze impossibili. Ogni voce include segnali linguistici precisi in italiano, inglese, tedesco e francese — il modello li cerca attivamente nel testo.

**Knowledge base dalle risposte passate**
Caricando i documenti con le risposte di Inpeco a gare precedenti, il sistema impara cosa l'azienda sa fare davvero — anche leggendo tra le righe del linguaggio diplomatico. Frasi come *"da confermare in fase di progetto"* o *"compatibile in linea di principio"* vengono riconosciute come segnali di incertezza e pesate di conseguenza nelle analisi future.

Il risultato è un'analisi che non è generica: conosce il contesto, i limiti reali e la storia dell'azienda.

---

## Output

Ogni analisi produce:

- Raccomandazione **Go / No-Go / Go with mitigation** con punteggio e motivazione
- Elenco showstopper rilevati con evidenza testuale
- Risk register con probabilità, impatto e score per ogni fattore
- Requisiti tecnici, commerciali e legali estratti dal documento
- Milestone e scadenze identificate
- Report Word formattato, scaricabile

Tre livelli di profondità: **Low** (~2 min), **Medium** (~4 min), **High** (~8 min).

---

## Moduli

| | |
|---|---|
| **Analyse Tender** | Upload documento, analisi GPT-4o, report interattivo |
| **Tender Library** | Storico di tutte le gare analizzate, esportabile CSV |
| **Risk Factors & Showstoppers** | Gestione del registro — aggiunta tramite linguaggio naturale, nessun JSON |
| **Past Bid Responses** | Caricamento risposte passate per arricchire la knowledge base |

---

## Stack

- Python · Streamlit
- OpenAI GPT-4o (API key propria — nessun dato condiviso con terze parti)
- PyMuPDF, pdfplumber, python-docx
- Tutto locale: nessun database, nessun cloud, file su disco

---

## Avvio

```bash
git clone https://github.com/lisaraco94-hub/tender-intake-assistant
cd tender-intake-assistant

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

---

## Note sul design

Il sistema migliora nel tempo. Ogni risposta passata caricata nella knowledge base affina la capacità di riconoscere pattern — capacità consolidate, aree di incertezza, linguaggio tipico delle situazioni limite. Non è un tool statico: è una base di conoscenza che cresce con l'uso.

---

[github.com/lisaraco94-hub/tender-intake-assistant](https://github.com/lisaraco94-hub/tender-intake-assistant)
