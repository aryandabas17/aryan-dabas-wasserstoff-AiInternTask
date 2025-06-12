# 🧠 Document Research & Theme Identification Chatbot

This project is a Streamlit-based AI assistant that reads images, documents (PDFs or screenshots), identifies the overall theme, and answers user questions about the content using local language models and embeddings — all without needing a GPU or paid APIs.

---

## ✨ Features

- 🔍 **Image & PDF Text Extraction** — OCR from screenshots or files
- 🧠 **Theme Summarization** — Extracts a concise theme from the full document
- 💬 **Interactive Q&A** — Ask questions like “What’s the work schedule?” or “What technologies are required?”
- 💾 **Local Embeddings** — Uses `bge-small-en` for vector search (retrieval-augmented)
- 🧑‍💻 **LLM-Powered Answers** — Uses lightweight local models like `gemma:2b` via [Ollama](https://ollama.com/)
- ✅ **No paid APIs required**
- 🎨 **Modern Streamlit UI** — Q&A history, download option, clear theme display

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Embeddings**: BAAI/bge-small-en (via HuggingFace)
- **Vector Database**: Chroma
- **LLMs**: 
  - Gemma 2B for Q&A
  - mistral:7b-instruct for summarizing tasks
- **OCR**: Tesseract via pytesseract
- **PDF Processing**: PyMuPDF (fitz)

## 📋 Requirements

- Python 3.8+
- Tesseract OCR installed on your system
- Ollama running locally for LLM inference

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/abhijith789/abhijith-k-r.git
   cd document-research-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - Windows: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

4. Install and run Ollama:
   - Follow instructions at [Ollama's website](https://ollama.ai/)
   - Pull required models:
     ```bash
     ollama pull gemma:2b
     ollama pull mistral:7b-instruct -- for enhanced performance (requires more computing resource)
     ```

5. Update the Tesseract path in `backend/app/services/ingest.py` if necessary:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path
   ```

## 📊 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Access the app in your browser (typically at http://localhost:8501)

3. Upload your documents (PDFs or images)

4. Wait for the processing to complete

5. (Optional) Click "Summarize Theme" to get a theme summary of your documents

6. Ask questions about your documents in the query box

## 📁 Project Structure

```
document-research-assistant/
├── backend/
│   └── app/
│       └── services/
│           ├── embed.py      # Document embedding functionality
│           ├── ingest.py     # PDF and image text extraction
│           ├── query.py      # Document querying functionality
│           └── summarize.py  # Document summarization
├── data/
│   ├── chroma_store/         # Vector database storage
│   ├── input_images/         # Temporary storage for uploaded files
│   └── text_outputs/         # Extracted text from documents
├── requirements.txt          # Python dependencies
├── streamlit_app.py          # Main Streamlit application
└── README.md                 # This file
```

## 🔧 Customization

- Change embedding models in `backend/app/services/embed.py`
- Adjust text splitting parameters in `embed_documents()` function
- Modify LLM models in `query.py` and `summarize.py`
- Update prompt templates for different response styles

## 🚨 Troubleshooting

- **Tesseract errors**: Ensure Tesseract is installed and the path is correctly set
- **Memory issues**: Reduce chunk sizes in the embedding process
- **LLM errors**: Make sure Ollama is running and the required models are pulled

## 📄 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
