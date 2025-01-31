from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Página principal
    path('process/', views.process_page, name='process'),  # Página de procesamiento
    path('thanks/', views.thanks_page, name='thanks_page'),  # Página de agradecimiento
    path('process-images/', views.process_images_view, name='process_images'),  # Procesar imágenes
]
