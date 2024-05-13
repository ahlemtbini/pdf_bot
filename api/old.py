import torch
import pdfplumber
from transformers import AlbertTokenizer, AlbertForQuestionAnswering, pipeline
from .models import TrainingDocument
print("GPU DISPO ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AlbertTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
model = AlbertForQuestionAnswering.from_pretrained("twmkn9/albert-base-v2-squad2").to(device)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

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
    result = qa_pipeline({
        "question": question,
        "context": context,
    }, max_answer_len=500)
    answer = result["answer"]

    # Format the response more interactively
    if "..." in answer or answer.strip() == "":
        return "Je suis désolé, je n'ai pas de réponse appropriée à votre question."
    return f"Voici ce que j'ai trouvé: {answer}"
