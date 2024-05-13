from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import TrainingDocument
from .serializers import TrainingDocumentSerializer, QuestionSerializer
from .utils import train_model, answer_question
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import json
import os
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, Trainer, TrainingArguments, pipeline
from django.http import JsonResponse
from .qa_dataset import QADataset
from torch.utils.data import DataLoader
from django.core.files.storage import default_storage



@method_decorator(csrf_exempt, name='dispatch')
class TrainAPIView(APIView):
    def post(self, request):
        serializer = TrainingDocumentSerializer(data=request.data)
        if serializer.is_valid():
            document = serializer.save()
            train_model(document)
            return Response({"message": "Training completed."}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
@method_decorator(csrf_exempt, name='dispatch')
class QuestionAPIView(APIView):
    def post(self, request):
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.data.get("question")
            document_id = serializer.data.get("document_id")
            try:
                document = TrainingDocument.objects.get(id=document_id, trained=True)
                answer = answer_question(question, document.text)
                return Response({"response": answer}, status=status.HTTP_200_OK)
            except TrainingDocument.DoesNotExist:
                return Response({"message": "Document not found or not trained."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
def train_and_evaluate_model(request):
    if request.method == 'POST' and request.FILES.get('json_file'):
        json_file = request.FILES['json_file']
        path = default_storage.save('temp.json', json_file)
        output_dir = "./fine_tuned_model"

        tokenizer = BertTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
        dataset = QADataset(path, tokenizer)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        model = BertForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

        # Test the model
        sample = dataset[0]
        result = qa_pipeline(question=sample['input_ids'], context=sample['attention_mask'])
        return JsonResponse({"answer": result})

    return JsonResponse({"error": "Invalid request"}, status=400)