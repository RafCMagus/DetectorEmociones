import os
import cv2
import dlib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir la clase VGG (modelo base)
class VGG(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(VGG, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 1 * 1, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Definir el metamodelo
class MetaModel(torch.nn.Module):
    def __init__(self, input_size, num_classes=7):
        super(MetaModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Cargar los pesos de los modelos base
def load_models(model_paths):
    models = []
    for model_path in model_paths:
        try:
            model = VGG(num_classes=7).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Error al cargar el modelo {model_path}: {e}")
    return models

# Cargar los pesos del metamodelo
def load_meta_model(meta_model_path):
    try:
        input_size = len(model_paths) * 7  # Número de modelos * número de clases
        meta_model = MetaModel(input_size=input_size, num_classes=7).to(device)
        meta_model.load_state_dict(torch.load(meta_model_path, map_location=device))
        meta_model.eval()
        return meta_model
    except Exception as e:
        print(f"Error al cargar el metamodelo: {e}")
        return None

# Transformaciones para las imágenes
image_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Mapeo de etiquetas de emociones
emotion_label_map = {
    0: 'Enojo (Angry)',
    1: 'Disgusto (Disgust)',
    2: 'Miedo (Fear)',
    3: 'Felicidad (Happy)',
    4: 'Tristeza (Sad)',
    5: 'Sorpresa (Surprise)',
    6: 'Neutral (Neutral)'
}

# Detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Procesar una imagen
def process_image(image_path, models, meta_model):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)
    print(f"{len(faces)} rostros detectados")

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = rgb_image[y:y+h, x:x+w]

        face_pil = Image.fromarray(face_img)
        face_tensor = image_transform(face_pil).unsqueeze(0).to(device)

        # Obtener predicciones de los modelos base
        probs = []
        for model in models:
            with torch.no_grad():
                output = model(face_tensor)
                probs.append(torch.softmax(output, dim=1).cpu().numpy())

        # Concatenar probabilidades y predecir con el metamodelo
        probs_stack = np.hstack(probs)
        probs_tensor = torch.tensor(probs_stack, dtype=torch.float32).to(device)

        with torch.no_grad():
            meta_output = meta_model(probs_tensor)
            meta_pred = torch.argmax(meta_output, dim=1).item()

        emotion_label = emotion_label_map.get(meta_pred, "Desconocido")
        print(f"Rostro detectado: {emotion_label}")

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    output_path = os.path.join("output", os.path.basename(image_path))
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Imagen procesada guardada en: {output_path}")

# Configuración de rutas
model_paths = [
    'D:/ModeloIA/BestMetaModelPesos/best_model_fold_1.pth',
    'D:/ModeloIA/BestMetaModelPesos/best_model_fold_2.pth',
    'D:/ModeloIA/BestMetaModelPesos/best_model_fold_3.pth',
    'D:/ModeloIA/BestMetaModelPesos/best_model_fold_4.pth',
    'D:/ModeloIA/BestMetaModelPesos/best_model_fold_5.pth'
]
meta_model_path = "D:/ModeloIA/BestMetaModelPesos/meta_model.pth"

# Cargar modelos base y metamodelo
models = load_models(model_paths)
meta_model = load_meta_model(meta_model_path)

if not models or not meta_model:
    print("Error crítico: No se pudieron cargar los modelos correctamente.")
    exit()

# Procesar una imagen
image_path = "D:/imagenes de prueba/i6.png"
process_image(image_path, models, meta_model)
