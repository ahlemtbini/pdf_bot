import pdfplumber
import requests
import os
from .models import TrainingDocument

# Set your Hugging Face API key here
HUGGING_FACE_API_KEY = "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"
API_URL = "https://api-inference.huggingface.co/models/deepset/bert-large-uncased-whole-word-masking-squad2"
headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def train_model(document: TrainingDocument):
    text = extract_text_from_pdf(document.file)
    document.text = text
    document.trained = True
    document.save()

def answer_question(question, context):
    payload = {"inputs": {"question": question, "context": context}}
    response = requests.post(API_URL, headers=headers, json=payload)

    # Check for errors
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.json().get('error', 'Unknown Error')}"

    result = response.json()
    answer = result.get("answer", "").strip()

    if not answer or "..." in answer or answer.lower().startswith("i'm sorry") or "pas de réponse" in answer.lower():
        return "Je suis désolé, je n'ai pas de réponse appropriée à votre question."
    return f"Voici ce que j'ai trouvé: {answer}"
