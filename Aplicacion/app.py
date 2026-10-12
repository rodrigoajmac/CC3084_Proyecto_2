"""Aplicación académica para comparar clasificadores cervicales preentrenados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError

from core.config import APP_DIR, load_app_config, load_model_configs
from core.data import find_manifest, load_manifest, summarize_manifest
from core.inference import PredictionError, predict
from core.metrics import load_all_metrics, metric_chart, confusion_figure
from core.model_loader import load_model, model_status


st.set_page_config(
    page_title="Explorador de modelos cervicales",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #f5f7fa; color: #162331; }
          [data-testid="stSidebar"] { background: #13283b; }
          [data-testid="stSidebar"] * { color: #f5f8fa !important; }
          .hero { padding: 1.4rem 1.6rem; background: linear-gradient(120deg,#163b58,#23627e);
                  border-radius: 14px; color: white; margin-bottom: 1rem; }
          .notice { border-left: 5px solid #d38b27; background: #fff8e8; padding: .8rem 1rem;
                    border-radius: 5px; color: #352a18; }
          .status-ok { color: #16734a; font-weight: 650; }
          .status-bad { color: #a33a35; font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_home(models) -> None:
    header("Explorador de modelos cervicales", "Prototipo académico de inferencia y comparación")
    st.markdown(
        '<div class="notice"><strong>No es una herramienta de diagnóstico.</strong> '
        "Las predicciones no sustituyen la evaluación de un radiólogo ni deben usarse "
        "para tomar decisiones clínicas.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Alcance")
    st.write(
        "La aplicación carga modelos ya entrenados desde archivos locales, aplica exactamente el "
        "preprocesamiento declarado y muestra sus salidas por separado. No entrena, ajusta ni genera modelos."
    )
    st.info(
        "El proyecto RSNA 2022 contiene volúmenes de tomografía computarizada axial, no radiografías "
        "laterales. Esta interfaz de imagen única solo es válida para modelos que hayan sido entrenados "
        "y documentados para una entrada 2D compatible."
    )
    st.subheader("Modelos configurados")
    if not models:
        st.warning("No hay modelos registrados. Edita `models.yaml` siguiendo `models.example.yaml`.")
        return
    for config in models:
        status = model_status(config)
        state = "Disponible" if status.ready else "No disponible"
        st.write(
            f"**{config.name}** — {config.framework} / {config.format} — "
            f"{state}: {status.message}"
        )


def render_data(app_config) -> None:
    header("Explorar datos", "Resumen y filtros del manifiesto disponible")
    manifest = find_manifest(app_config)
    if manifest is None:
        st.info(
            "No se encontró un manifiesto CSV. Configura `manifest_path` en `app_config.yaml` o coloca "
            "el archivo en la ruta indicada. No se muestran conteos hasta disponer de datos reales."
        )
        return
    try:
        frame = load_manifest(manifest)
    except Exception as exc:
        st.error(f"No se pudo leer {manifest.name}: {exc}")
        return
    summary = summarize_manifest(frame)
    st.caption(f"Fuente: {manifest}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observaciones", f"{len(frame):,}")
    c2.metric("Variables", len(frame.columns))
    c3.metric("Valores ausentes", f"{int(frame.isna().sum().sum()):,}")
    if summary is not None:
        label_name, counts = summary
        fig = px.bar(
            x=[str(value) for value in counts.index], y=counts.values,
            labels={"x": label_name, "y": "Cantidad"}, title=f"Distribución de {label_name}",
            color_discrete_sequence=["#23627e"],
        )
        st.plotly_chart(fig, use_container_width=True)
    query = st.text_input("Filtrar filas (búsqueda de texto en todas las variables)")
    shown = frame
    if query:
        mask = frame.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        shown = frame.loc[mask]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    with st.expander("Variables disponibles"):
        st.dataframe(
            pd.DataFrame({"variable": frame.columns, "tipo": frame.dtypes.astype(str).values,
                          "nulos": frame.isna().sum().values}),
            hide_index=True, use_container_width=True,
        )


def render_analysis(models, app_config) -> None:
    header("Analizar imagen", "Inferencia con modelos preentrenados configurados")
    max_mb = int(app_config.get("max_upload_mb", 20))
    max_pixels = int(app_config.get("max_image_pixels", 40_000_000))
    uploaded = st.file_uploader(
        f"Carga una imagen PNG o JPEG (máximo {max_mb} MB)", type=["png", "jpg", "jpeg"]
    )
    if uploaded is None:
        st.info("La imagen se procesa únicamente en memoria y no se guarda en disco.")
        return
    if uploaded.size > max_mb * 1024 * 1024:
        st.error(f"El archivo supera el límite de {max_mb} MB.")
        return
    try:
        image = Image.open(uploaded)
        if image.width * image.height > max_pixels:
            st.error(f"La imagen supera el límite de {max_pixels:,} píxeles.")
            return
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        st.error(f"El archivo no es una imagen válida: {exc}")
        return
    if image.width < 16 or image.height < 16:
        st.error("La imagen es demasiado pequeña para una inferencia válida (mínimo 16 × 16 píxeles).")
        return
    st.image(image, caption=f"Vista previa — {image.width} × {image.height} px", width=520)
    confirmed = st.checkbox(
        "Confirmo que la imagen corresponde al tipo de estudio y proyección esperados por el modelo."
    )
    st.caption("La aplicación no verifica automáticamente la anatomía ni la proyección.")
    ready = [item for item in models if model_status(item).ready]
    if not ready:
        st.warning("No hay modelos listos para inferencia. Revisa la configuración y los archivos.")
        return
    options = ["Todos los modelos disponibles"] + [item.name for item in ready]
    selected = st.selectbox("Modelo", options)
    if st.button("Ejecutar análisis", type="primary", disabled=not confirmed):
        chosen = ready if selected == options[0] else [item for item in ready if item.name == selected]
        results = []
        for config in chosen:
            try:
                with st.spinner(f"Ejecutando {config.name}…"):
                    runtime = load_model(config.cache_key, config.as_dict())
                    result = predict(runtime, config, image)
                results.append(result)
            except Exception as exc:
                st.error(f"{config.name}: {exc}")
        if results:
            st.subheader("Resultados")
            frame = pd.DataFrame([item.as_row() for item in results])
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.caption(
                "Las puntuaciones son salidas del modelo. Solo se denominan probabilidades cuando la "
                "configuración declara explícitamente una salida probabilística; no son riesgo clínico."
            )
            fig = px.bar(frame, x="Modelo", y="Puntuación de fractura", color="Clase predicha",
                         range_y=[0, 1] if all(item.bounded_score for item in results) else None)
            st.plotly_chart(fig, use_container_width=True)


def render_performance(models) -> None:
    header("Rendimiento", "Métricas declaradas del conjunto de prueba")
    show = st.toggle("Mostrar métricas del conjunto de prueba", value=False)
    if not show:
        st.info("Activa el control para cargar y mostrar las métricas configuradas.")
        return
    records, matrices, messages = load_all_metrics(models)
    for message in messages:
        st.info(message)
    if records.empty:
        st.warning("No hay métricas de prueba disponibles. Agrégalas y configura `metrics_path` por modelo.")
        return
    st.plotly_chart(metric_chart(records), use_container_width=True)
    st.dataframe(records, use_container_width=True, hide_index=True)
    for model_name, matrix in matrices.items():
        st.plotly_chart(confusion_figure(model_name, matrix), use_container_width=True)


def main() -> None:
    inject_styles()
    try:
        app_config = load_app_config()
        models, config_errors = load_model_configs()
    except Exception as exc:
        st.error(f"Error de configuración: {exc}")
        st.stop()
    with st.sidebar:
        st.title("Navegación")
        page = st.radio("Sección", ["Inicio", "Explorar datos", "Analizar imagen", "Rendimiento"])
        st.divider()
        st.caption("Uso académico · Sin almacenamiento de imágenes")
        if config_errors:
            with st.expander("Errores de configuración"):
                for error in config_errors:
                    st.error(error)
    if page == "Inicio":
        render_home(models)
    elif page == "Explorar datos":
        render_data(app_config)
    elif page == "Analizar imagen":
        render_analysis(models, app_config)
    else:
        render_performance(models)


if __name__ == "__main__":
    main()
