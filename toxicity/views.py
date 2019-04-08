from django.shortcuts import render
from .models import Images
# Create your views here.

def index(request):
    if request.method=='POST':
        image = Images(
            image = request.FILES.get('image')

        )
        image.save()
        images = request.FILES.get('image')
    return render(request,"toxicity/index.html")