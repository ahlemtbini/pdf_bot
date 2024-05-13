import os
from .models import TrainingDocument
import some_large_language_model as llm # Remplacez par votre modèle LLM

def train_model(document: TrainingDocument):
    pdf_file_path = document.file.path
    llm.train_on_pdf(pdf_file_path)
    document.trained = True
    document.save()
