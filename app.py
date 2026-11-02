import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.classifier import DamageClassifier
from src.config import DAMAGE_MODEL_ID, PART_MODEL_PATH
from src.detector import CarPartDetector
from src.pipeline import CarDamagePipeline
from src.visualization import draw_detections


@st.cache_resource
def load_pipeline() -> CarDamagePipeline:
    return CarDamagePipeline(
        CarPartDetector(PART_MODEL_PATH),
        DamageClassifier(DAMAGE_MODEL_ID),
    )


st.set_page_config(
    page_title="Car Damage MVP",
    layout="wide"
)

st.title("🚗 Detección de piezas y daños")
st.write("Segmentación de piezas con YOLOv8 y clasificación de daños.")
st.caption(
    "El clasificador siempre elige una de sus seis clases de daño; "
    "no incluye una categoría de «sin daño». Revisá cada predicción."
)

uploaded_file = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if not PART_MODEL_PATH.is_file():
        st.error(
            f"No se encontró el modelo de piezas: {PART_MODEL_PATH}. "
            "Entrenalo primero ejecutando `python train_detector.py`."
        )
        st.stop()

    image = cv2.imdecode(
        np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        st.error("No se pudo leer la imagen cargada.")
        st.stop()

    with st.spinner("Segmentando piezas y clasificando daños..."):
        pipeline = load_pipeline()
        detections = pipeline.analyze(image)

    annotated = draw_detections(
        image,
        detections
    )

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Imagen Procesada")
        st.image(
            annotated,
            use_container_width=True
        )

    with right:
        st.subheader("Piezas y daños estimados")

        if detections:
            df = pd.DataFrame(detections)
            st.dataframe(
                df[
                    [
                        "part",
                        "confidence",
                        "damage",
                        "damage_confidence",
                    ]
                ].rename(
                    columns={
                        "part": "Pieza",
                        "confidence": "Confianza pieza",
                        "damage": "Daño estimado",
                        "damage_confidence": "Confianza daño",
                    }
                ),
                use_container_width=True
            )
        else:
            st.warning("No se detectaron piezas del automóvil.")
