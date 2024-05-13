from django.db import models

class TrainingDocument(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    text = models.TextField(blank=True)
    trained = models.BooleanField(default=False)
    
    def __str__(self):
        return self.title
