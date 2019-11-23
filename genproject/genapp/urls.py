from django.urls import path
from . import views

app_name = 'genapp'
urlpatterns = [
    path('', views.index, name='index'),
]