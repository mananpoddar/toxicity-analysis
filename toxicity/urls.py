from django.conf.urls import url
from toxicity import views
app_name = "toxicity"

urlpatterns = [
    url(r'^$', views.index, name='index'),
]