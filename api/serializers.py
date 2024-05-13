from rest_framework import serializers
from .models import TrainingDocument

class TrainingDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingDocument
        fields = ['id', 'title', 'file', 'text', 'trained']

class QuestionSerializer(serializers.Serializer):
    question = serializers.CharField()
    document_id = serializers.IntegerField()
