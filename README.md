# ProyectoLOL — Detección de Skins de League of Legends con YOLO

## Descripción del proyecto

Este proyecto implementa un sistema de **detección de objetos** capaz de identificar y localizar *skins* de campeones de League of Legends en imágenes y videos. El modelo se entrena utilizando **YOLO26n** (Ultralytics) sobre un dataset personalizado anotado y gestionado a través de **Roboflow**.

El flujo completo del proyecto incluye:

1. Descarga automática del dataset anotado desde Roboflow.
2. Entrenamiento de un modelo YOLO26n preentrenado, ajustado (*fine-tuned*) sobre el dataset de skins.
3. Evaluación del modelo mediante métricas estándar de detección de objetos: `mAP50`, `mAP50-95`, Precisión, Recall, curvas F1/Precision/Recall y matriz de confusión.
4. Inferencia sobre imágenes individuales, mostrando las cajas delimitadoras (*bounding boxes*), la clase detectada y el nivel de confianza.
5. Inferencia sobre **video**, procesando frame por frame y generando un video de salida con las detecciones anotadas.

El entrenamiento y la inferencia están configurados para ejecutarse utilizando la **GPU local (NVIDIA, vía CUDA)**, acelerando significativamente el proceso frente a CPU.

---

## Librerías necesarias

El proyecto requiere **Python 3.10, 3.11 o 3.12** y las siguientes librerías:

| Librería | Uso en el proyecto |
|---|---|
| `torch`, `torchvision`, `torchaudio` (con soporte CUDA) | Motor de deep learning y aceleración por GPU |
| `ultralytics` | Framework de YOLO26 (entrenamiento, evaluación e inferencia) |
| `roboflow` | Descarga y gestión del dataset anotado |
| `opencv-python` (`cv2`) | Lectura/escritura de imágenes y video, dibujo de detecciones |
| `matplotlib` | Visualización de imágenes, métricas y curvas de evaluación |
| `numpy` | Manejo de arreglos numéricos |

### Instalación

```bash


#1. Clonar el repositorio del github
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio


# 2. Crear y activar un entorno virtual
#2.1 abrir terminal y crear entorno
python -m venv venv
venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source venv/bin/activate       # Linux / macOS

# 3. Instalar ipykernel primero (para que VS Code detecte el venv en caso de no detectarlo)
python -m pip install --upgrade pip
python -m pip install ipykernel


# 4. Instalar PyTorch con soporte CUDA (ajusta cuXXX según tu driver de GPU, ver https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Instalar el resto de dependencias
pip install ultralytics roboflow opencv-python matplotlib numpy
```

### Verificar que la GPU está disponible

```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Ejemplo de inferencia

Pueden encontrar el ejemplo de inferencia en la carpeta docs. Donde se muestra el codigo y un ejemplo de una foto que salio de resultado



---

## Estructura del proyecto

```
Proyecto_Trabajo_Lol/
├── ProyectoLOL (2).ipynb        # Notebook principal (entrenamiento, evaluación, inferencia)
├── yolo26n.pt                    # Pesos preentrenados de YOLO26n (punto de partida)
├── Deteccion-Skins-5/            # Dataset descargado desde Roboflow
├── runs/
│   └── detect/
│       ├── train-*/weights/best.pt   # Modelo entrenado (mejores pesos)
│       └── val/                      # Métricas, curvas y visualizaciones de validación
├── docs/
│    ├──Captura de pantalla......png    # Imagen de ejemplo con la inferencia realizada
     └──Captura de pantalla......png    # Imagen de ejemplo con la inferencia realizada
│                       
└── venv/                         # Entorno virtual de Python
```

---

## Uso rápido

1. Abrir `ProyectoLOL (2).ipynb` en VS Code (o Jupyter).
2. Seleccionar el kernel del entorno virtual (`venv`).
3. Ejecutar las celdas en orden: descarga de dataset → entrenamiento → evaluación → inferencia en imagen o video.

Para correr inferencia sobre un video propio:

```python
model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='video_prueba.mp4', save=True, device=0)
```
