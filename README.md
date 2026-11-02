# Car Damage MVP

Aplicación web en Streamlit para:

1. Segmentar piezas de automóviles con YOLOv8-seg, ajustado con el dataset
   Carparts-Seg.
2. Clasificar el daño estimado en el recorte de cada pieza con el modelo
   `beingamit99/car_damage_detection` de Hugging Face.
3. Mostrar las piezas, las máscaras, las cajas, las etiquetas y sus confianzas.

El flujo detallado está en [PIPELINE.md](./PIPELINE.md).

## Requisitos

- Windows, macOS o Linux.
- Python 3.10 o posterior.
- [uv](https://docs.astral.sh/uv/).
- Conexión a Internet la primera vez que se use el clasificador: Transformers
  descargará el modelo desde Hugging Face.

## Preparar el entorno con uv

Desde la carpeta del proyecto:

```powershell
uv sync
```

`uv sync` crea `.venv`, instala las dependencias de `pyproject.toml` y usa
`uv.lock` para resolver versiones reproducibles. Para activar el entorno en
PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si la descarga de paquetes falla por certificados TLS en Windows, probá:

```powershell
uv --system-certs sync
```

También se puede ejecutar sin activar el entorno, usando `uv run`.

## Ejecutar la aplicación

El modelo de piezas ya entrenado está en
`models/yolo/carparts-seg.pt`, por lo que no hace falta volver a entrenarlo para
usar la app:

```powershell
uv run streamlit run app.py
```

En el primer análisis, el clasificador y su procesador de imágenes se descargan
desde Hugging Face. Las ejecuciones posteriores pueden reutilizar la caché
local.

## Entrenar o reanudar el detector

El entrenamiento es opcional si se conserva el modelo incluido. Para iniciar
un entrenamiento nuevo de YOLOv8n-seg sobre Carparts-Seg:

```powershell
uv run python train_detector.py
```

El entrenamiento está configurado para 25 épocas y tamaño de imagen 640.
Ultralytics descarga el dataset y los pesos base si no están disponibles en su
caché. Los resultados se guardan bajo `runs/segment/carparts-seg/`; al finalizar
se copia el mejor checkpoint a `models/yolo/carparts-seg.pt`.

Para reanudar un entrenamiento interrumpido, conservá la carpeta de resultados
y ejecutá:

```powershell
uv run python train_detector.py --resume
```

La reanudación requiere el checkpoint `runs/segment/carparts-seg/weights/last.pt`.

## Pruebas

Ejecutá las pruebas unitarias con:

```powershell
uv run python -m unittest discover -s tests -v
```

## Componentes principales

| Ruta | Función |
| --- | --- |
| `app.py` | Interfaz Streamlit, carga de imágenes y presentación de resultados. |
| `src/detector.py` | Carga YOLO y devuelve piezas, cajas y máscaras. |
| `src/classifier.py` | Carga desde Hugging Face y predice una clase de daño. |
| `src/pipeline.py` | Recorta cada pieza detectada y encadena ambos modelos. |
| `src/visualization.py` | Dibuja máscaras, cajas y etiquetas sobre la imagen. |
| `src/config.py` | Rutas y nombres de los modelos/dataset. |
| `train_detector.py` | Entrena o reanuda el detector de segmentación. |
| `models/yolo/carparts-seg.pt` | Pesos entrenados usados directamente por la app. |
| `tests/` | Pruebas unitarias del detector, pipeline y reanudación. |
| `PIPELINE.md` | Diagrama y descripción del flujo de entrenamiento e inferencia. |

## Estado y limitaciones

- El modelo de segmentación se entrenó durante 25 épocas con `imgsz=640`.
  En la última época, las métricas de validación de máscaras registradas en
  `runs/segment/carparts-seg/results.csv` fueron `mAP50(M)=0.66416` y
  `mAP50-95(M)=0.53126`. Son métricas del conjunto de validación, no una
  evaluación independiente del conjunto de prueba.
- Carparts-Seg incluye 23 clases de piezas.
- El clasificador estima una de seis etiquetas: `Crack`, `Scratch`, `Tire Flat`,
  `Dent`, `Glass Shatter` o `Lamp Broken`. No tiene una clase «sin daño», así
  que puede asignar una etiqueta de daño incluso a una pieza sana.
- El pipeline clasifica el recorte rectangular de cada pieza (su caja
  delimitadora); la máscara se usa para la visualización y no para excluir el
  fondo del recorte.
- Las predicciones son estimaciones y deben revisarse antes de usarse para
  decisiones de reparación, seguros o seguridad.

## Publicar en GitHub

El `.gitignore` excluye el entorno virtual, el dataset descargado, las
imágenes locales de `testcases/`, las ejecuciones/checkpoints de entrenamiento
y otros archivos generados. Mantiene incluido `models/yolo/carparts-seg.pt`
(aprox. 6.7 MB) para que un clon pueda ejecutar la app sin entrenar de nuevo.

Antes de publicar, revisá las licencias y condiciones de redistribución del
modelo entrenado, sus modelos base, el dataset y el clasificador de Hugging Face.
