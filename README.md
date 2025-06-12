# 🧠 Document Research & Theme Identification Chatbot

This project is part of the **Wasserstoff AI Internship Task**. It’s an AI-powered assistant that performs theme summarization and Q&A from documents like PDFs and screenshots — fully locally using OCR, LLMs, and embeddings.

---

## ✨ Features

- 🔍 **OCR from PDFs & Images** (Tesseract)
- 🧠 **Theme Summarization** (via `mistral:7b-instruct`)
- 💬 **Ask Questions** like “What skills are listed?” or “What is the project overview?”
- 📚 **Local Embeddings & Retrieval** (`BAAI/bge-small-en` + ChromaDB)
- 🤖 **LLM-powered Answers** using Ollama's `gemma:2b`
- 🎨 **Modern Streamlit UI** with Q&A history, theme preview, and document upload
- ✅ **Runs without paid APIs or GPU**

---

## 🛠️ Tech Stack

| Component       | Tool / Model                      |
|----------------|------------------------------------|
| Frontend       | Streamlit                          |
| Backend        | Python                             |
| Embeddings     | `BAAI/bge-small-en` (HuggingFace)  |
| Vector DB      | ChromaDB                           |
| OCR            | Tesseract (`pytesseract`)          |
| LLM (Q&A)      | `gemma:2b` (via Ollama)            |
| LLM (Summary)  | `mistral:7b-instruct` (via Ollama) |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/aryandabas17/aryan-dabas-wasserstoff-AiInternTask.git
cd aryan-dabas-wasserstoff-AiInternTask

### 2. Create and activate virtual environment (Windows)
```bash
python -m venv env
env\Scripts\activate

### 3. Install dependencies
```bash
pip install -r aryan-dabas/requirements.txt

### 4. Install Tesseract OCR
```bash
Download & install: Tesseract GitHub Release
Then update the following line in ingest.py:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

### 5. Install & Run Ollama
```bash
ollama run gemma:2b
ollama pull mistral:7b-instruct

### 6. Run the Streamlit app
```bash
streamlit run aryan-dabas/streamlit_app.py

### 📍Note: Make sure your ollama server is running, and you’ve pulled the required models before opening the app.

