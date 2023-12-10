from django.urls import path
from matcher.views import PredictGenreView

urlpatterns = [
    path('predict_genre/', PredictGenreView.as_view(), name='predict_genre'),
    # Add other URL patterns as needed
]
