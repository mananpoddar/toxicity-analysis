from django.shortcuts import render
from .models import Images
import cv2
import sys
import pytesseract
# Create your views here.

def index(request):
    if request.method=='POST':
        image = Images(
            image = request.FILES.get('image')

        )
        image.save()
        images = request.FILES.get('image')
        text = tesseract(images)
        
    return render(request,"toxicity/index.html")

def tesseract(images):
    config = ('-l eng --oem 1 --psm 3')
    path = "./media/images/"+str(images)
    images = path
    im = cv2.imread(images,cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(im,config=config)
    return  text
