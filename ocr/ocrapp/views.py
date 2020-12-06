from django.shortcuts import render
from django.http import HttpResponse
import csv
import cv2
import pytesseract

# Create your views here.
def index(request):
    context = {}
    return render(request, 'ocrapp/index.html', context)
