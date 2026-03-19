# AI-credit-risk-analyzer

## Overview
This project demonstrates an **AI-powered credit assessment system** that automates the extraction and analysis of financial statements.

It combines:
- 📄 OCR (Optical Character Recognition)
- 🤖 LLM-based financial data standardisation
- 📊 Financial ratio computation
- 💡 Analyst-friendly insights generation

The goal is to **reduce manual effort in credit underwriting** and improve efficiency in financial analysis workflows.

---

## Key Features

- Upload multiple financial documents (PDF, images)
- OCR-based text extraction using Tesseract
-  Automatic unit detection (Thousands / Millions)
- LLM-based financial field extraction (via Ollama)
-  Editable financial values before computation
- Automated financial ratio calculation:
  - Current Ratio
  - Quick Ratio
  - Debt-to-Equity
  - Interest Coverage
  - DSCR
- Simple, explainable insights for analysts

---

## Architecture

```text
[ Uploaded Documents ]
        ↓
[ OCR Extraction (Tesseract + Poppler) ]
        ↓
[ LLM Standardisation (Ollama - phi3) ]
        ↓
[ Structured Financial Data ]
        ↓
[ Ratio Calculation Engine ]
        ↓
[ Insights + Analyst UI (Streamlit) ]
```
## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit  
- **OCR:** Tesseract OCR + Poppler  
- **LLM:** Ollama (phi3 model)  
- **Backend:** Python  
- **Data Processing:** Pandas  

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/govanzz/AI-credit-risk-analyzer.git
cd AI-credit-risk-analyzer

```
### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ System Dependencies

You must install:

- **Tesseract OCR**  
  Download: https://github.com/tesseract-ocr/tesseract  

- **Poppler**  
  Download: https://github.com/oschwartz10612/poppler-windows  

---

## ▶️ Run the App

```bash
streamlit run app.py
```

## Example Workflow

1. Upload:
   - Balance Sheet  
   - Income Statement  

2. System will:
   - Extract text via OCR  
   - Detect number scale  
   - Send to LLM for structuring  

3. Review extracted values  

4. (Optional) Correct values  

5. View ratios + insights  

---

## Limitations

- Requires local setup (Tesseract, Poppler, Ollama)  
- OCR accuracy depends on document quality  
- LLM extraction may require manual correction in edge cases  

---

## Future Improvements

- Replace local LLM with hosted API (OpenAI / Azure)  
- Add RAG for financial context understanding  
- Deploy on Streamlit Cloud  
- Add credit scoring model  
- Support more financial formats  

---

## Use Case

This system is designed for:

- Credit analysts  
- Banking workflows  
- SME loan underwriting  
- Financial document processing automation  

---







