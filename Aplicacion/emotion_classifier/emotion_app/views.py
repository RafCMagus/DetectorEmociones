from django.shortcuts import render
import os
from .services.image_processing import process_image, has_processed_images
from django.http import JsonResponse
from django.conf import settings



# Página principal
def home(request):
    """Renderiza la página principal."""
    return render(request, 'emotion_app/home.html')

# Página de procesamiento
def process_page(request):
    """Renderiza la página de procesamiento, verificando si hay imágenes en la galería."""
    gallery_available = has_processed_images()
    return render(request, 'emotion_app/process.html', {'gallery_available': gallery_available})

# Procesar imágenes seleccionadas
# Procesar imágenes seleccionadas
def process_images_view(request):
    """Procesa una o más imágenes seleccionadas por el usuario."""
    if request.method == 'POST' and request.FILES.getlist('images'):
        uploaded_images = request.FILES.getlist('images')
        results = []

        for image_file in uploaded_images:
            # Guardar la imagen en un archivo temporal
            temp_image_path = os.path.join("temp_uploads", image_file.name)
            os.makedirs("temp_uploads", exist_ok=True)

            with open(temp_image_path, "wb") as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Procesar la imagen con la ruta correcta
            processed_image_path = process_image(temp_image_path)  
            results.append(processed_image_path)

            # Opcional: Eliminar la imagen después de procesarla
            os.remove(temp_image_path)

        message = f"Se procesaron {len(results)} imágenes correctamente."
        return render(request, 'emotion_app/process.html', {'message': message, 'results': results})

    return render(request, 'emotion_app/process.html', {'message': 'Por favor selecciona al menos una imagen para procesar.'})

# Página de agradecimiento
def thanks_page(request):
    """Renderiza la página de agradecimiento al salir de la aplicación."""
    return render(request, 'emotion_app/thanks.html')

def get_processed_images(request):
    """Devuelve una lista de imágenes en la carpeta output."""
    output_folder = os.path.join(settings.STATICFILES_DIRS[0], "output")
    images = [img for img in os.listdir(output_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    return JsonResponse({"images": images})