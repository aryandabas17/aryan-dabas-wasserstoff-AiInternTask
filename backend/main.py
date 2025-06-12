from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.app.services import ingest, summarize, query
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/input_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    text = ingest.extract_text_from_file(filepath)
    with open(f"data/text_outputs/{file.filename}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return {"message": "File uploaded and text extracted"}

@app.get("/theme")
def get_theme():
    return {"summary": summarize.summarize_theme()}

@app.post("/question")
def ask_question(payload: dict):
    question = payload.get("question", "")
    answer = query.ask_question(question)
    return {"answer": answer}
