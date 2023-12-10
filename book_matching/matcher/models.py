from django.db import models

# Create your models here.
# book_matching_app/models.py


class Student(models.Model):
    conscientiousness = models.FloatField()
    openness = models.FloatField()
    predicted_genre = models.CharField(max_length=50)
    # Add more traits if needed


