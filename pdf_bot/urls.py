from django.urls import path
from api.views import TrainAPIView, QuestionAPIView,train_and_evaluate_model

urlpatterns = [
    path('train/', TrainAPIView.as_view(), name='train'),
    path('question/', QuestionAPIView.as_view(), name='question'),
    path('trainEvaluate/', train_and_evaluate_model, name='train_and_evaluate'),

]
