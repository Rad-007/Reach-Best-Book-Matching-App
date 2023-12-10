from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse,FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
from matcher.models import Student
from matcher.predict_genre import predict_book_genre  # Replace 'your_module' with the actual module containing the predict_genre function
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import base64
from io import BytesIO

@method_decorator(csrf_exempt, name='dispatch')
class PredictGenreView(View):
    def post(self, request, *args, **kwargs):
        data = json.loads(request.body.decode('utf-8'))

        #criteria1 = list(ast.literal_eval(data.get('criteria1', [])))
        #criteria2 = list(ast.literal_eval(data.get('criteria2', [])))
        criteria1=data.get('criteria1', [])
        criteria2=data.get('criteria2', [])
        #criteria1 = criteria1[1:-1].split(',')
        #criteria2 = criteria2[1:-1].split(',')

        print(criteria1)

        # Calculate the average of each tuple and convert to integer
        conscientiousness = int((np.mean([float(i) for i in criteria1])*2) -1)
        openness = int((np.mean([float(i) for i in criteria2])*2) -1)

        print(conscientiousness,openness)

        # Use the predict_genre function to get the genre prediction
        predicted_genre = predict_book_genre(int(conscientiousness), int(openness))

        # Store data in the database
        student = Student.objects.create(conscientiousness=conscientiousness, openness=openness, predicted_genre=predicted_genre)

        #create plot
        criteria1 = [int(i) for i in criteria1]
        criteria2 = [int(i) for i in criteria2]

        plt.figure()
        plt.plot(criteria1, 'o-', label='Conscientiousness')  # Plot criteria1 as a line with markers
        plt.plot(criteria2, label='Openess')  # Plot criteria2 as a line
        plt.legend()  # Add a legend


        buf = BytesIO()
        plt.savefig(buf, format='png')
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        # Open the file in a FileResponse
        #response = FileResponse(open('plot.png', 'rb'))

        # Delete the file after sending the response
        #os.remove('plot.png')






        return JsonResponse({'predicted_genre': predicted_genre, 'student_id': student.id,'plot': image_base64})
