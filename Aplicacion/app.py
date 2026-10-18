"""Aplicación académica para comparar clasificadores cervicales preentrenados."""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from core.config import load_app_config, load_model_configs
from core.data import find_manifest, load_manifest, summarize_manifest
from core.inference import predict
from core.metrics import MODEL_COLORS, load_all_metrics, metric_chart, confusion_figure
from core.model_loader import load_model, model_status
from core.preprocessing import prepare_dicom


CLASS_COLORS = {"Sin fractura": "#145A73", "Fractura": "#BD6430"}


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
    st.info("Los modelos analizan un corte axial de tomografía CT en formato DICOM. "
            "La app aplica automáticamente la misma ventana ósea y tamaño usados al entrenar.")
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
    st.caption("Cada fila representa un corte usado para modelado; las redes usan los píxeles CT, "
               "mientras estas columnas describen etiquetas y metadatos del estudio.")
    filtered = frame.copy()
    if "label" in frame:
        labels = st.multiselect("Clase del corte", ["Sin fractura", "Fractura"],
                                default=["Sin fractura", "Fractura"])
        filtered = filtered.loc[filtered["label"].map({0: "Sin fractura", 1: "Fractura"}).isin(labels)]
    if "tipo" in frame:
        kinds = sorted(frame["tipo"].dropna().astype(str).unique())
        chosen_kinds = st.multiselect("Origen de la muestra", kinds, default=kinds)
        filtered = filtered.loc[filtered["tipo"].isin(chosen_kinds)]
    if filtered.empty:
        st.info("Los filtros no dejaron cortes para mostrar.")
        return
    if summary is not None and "label" in filtered:
        counts = filtered["label"].value_counts().reindex([0, 1], fill_value=0)
        distribution = pd.DataFrame({"Clase": ["Sin fractura", "Fractura"],
                                     "Cortes": counts.values})
        fig = px.bar(distribution, x="Clase", y="Cortes", color="Clase",
                     color_discrete_map=CLASS_COLORS, title="Balance de clases en los cortes filtrados")
        st.plotly_chart(fig, width="stretch")
        st.caption("El balance refleja el muestreo usado para entrenar; no representa la prevalencia clínica.")
    numeric = [column for column in ("corte", "n_slices", "slice_thickness", "pixel_spacing_x",
                                     "z_extent", "rows", "cols") if column in filtered]
    if numeric and "label" in filtered:
        selected_variable = st.selectbox("Explorar relación con la clase", numeric,
                                          format_func=lambda name: {
                                              "corte": "Número de corte", "n_slices": "Cortes por estudio",
                                              "slice_thickness": "Grosor de corte (mm)",
                                              "pixel_spacing_x": "Espaciado de píxel (mm)",
                                              "z_extent": "Extensión axial (mm)", "rows": "Filas", "cols": "Columnas",
                                          }.get(name, name))
        plot_data = filtered[[selected_variable, "label"]].copy()
        plot_data[selected_variable] = pd.to_numeric(plot_data[selected_variable], errors="coerce")
        plot_data = plot_data.dropna()
        plot_data["Clase"] = plot_data["label"].map({0: "Sin fractura", 1: "Fractura"})
        fig = px.histogram(plot_data, x=selected_variable, color="Clase", barmode="overlay",
                           histnorm="probability density", opacity=0.7, nbins=35,
                           color_discrete_map=CLASS_COLORS,
                           title=f"Distribución de {selected_variable} por clase")
        st.plotly_chart(fig, width="stretch")
        st.caption("Las distribuciones se normalizan por clase para comparar formas. "
                   "Una asociación aquí no demuestra que el metadato cause una fractura.")
    query = st.text_input("Filtrar filas (búsqueda de texto en todas las variables)")
    shown = filtered
    if query:
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False, regex=False)).any(axis=1)
        shown = filtered.loc[mask]
    st.caption(f"Mostrando {len(shown):,} de {len(frame):,} cortes")
    st.dataframe(shown, width="stretch", hide_index=True)
    with st.expander("Variables disponibles"):
        st.dataframe(
            pd.DataFrame({"variable": frame.columns, "tipo": frame.dtypes.astype(str).values,
                          "nulos": frame.isna().sum().values}),
            hide_index=True, width="stretch",
        )


def render_analysis(models, app_config) -> None:
    header("Analizar corte CT", "Clasificación de un corte DICOM con los modelos entrenados")
    max_mb = int(app_config.get("max_upload_mb", 20))
    max_pixels = int(app_config.get("max_image_pixels", 40_000_000))
    uploaded = st.file_uploader(
        f"Carga un corte CT DICOM .dcm (máximo {max_mb} MB)", type=["dcm", "dicom"]
    )
    if uploaded is None:
        st.info("Se convertirá a HU, se aplicará ventana ósea (400/1800) y se ajustará a 224 × 224 "
                "automáticamente. El corte se procesa en memoria.")
        return
    if uploaded.size > max_mb * 1024 * 1024:
        st.error(f"El archivo supera el límite de {max_mb} MB.")
        return
    try:
        image = prepare_dicom(uploaded.getvalue(), max_pixels)
    except Exception as exc:
        st.error(f"No se pudo preparar el DICOM: {exc}. Si está comprimido, instala pylibjpeg y pylibjpeg-libjpeg.")
        return
    st.image(image, caption="Corte en ventana ósea preparado para los modelos", width=520)
    st.caption("Clasificación exploratoria de un corte. El sistema no verifica automáticamente la anatomía "
               "ni resume un estudio completo.")
    with st.expander("Cómo quedó la entrada tras el preprocesamiento"):
        intensities = pd.DataFrame({"Intensidad": np.asarray(image).ravel()})
        fig = px.histogram(intensities, x="Intensidad", nbins=40,
                           color_discrete_sequence=["#145A73"],
                           title="Distribución de intensidades en ventana ósea")
        fig.update_xaxes(range=[0, 255])
        st.plotly_chart(fig, width="stretch")
        st.caption("Los modelos reciben estos valores de 0 a 255. La CNN usa un canal; "
                   "MobileNetV2 y EfficientNetB0 reciben el mismo corte repetido en tres canales.")
    ready = [item for item in models if model_status(item).ready]
    if not ready:
        st.warning("No hay modelos listos para inferencia. Revisa la configuración y los archivos.")
        return
    options = ["Todos los modelos disponibles"] + [item.name for item in ready]
    selected = st.selectbox("Modelo", options)
    if st.button("Ejecutar análisis", type="primary"):
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
            thresholds = {item.name: item.threshold for item in chosen}
            frame["Umbral de clasificación"] = frame["Modelo"].map(thresholds)
            st.dataframe(frame, width="stretch", hide_index=True)
            st.caption(
                "Las puntuaciones son salidas del modelo. Solo se denominan probabilidades cuando la "
                "configuración declara explícitamente una salida probabilística; no son riesgo clínico."
            )
            fig = px.bar(frame, x="Modelo", y="Puntuación de fractura", color="Clase predicha",
                         color_discrete_map={"sin_fractura": CLASS_COLORS["Sin fractura"],
                                             "fractura": CLASS_COLORS["Fractura"]},
                         range_y=[0, 1] if all(item.bounded_score for item in results) else None)
            st.plotly_chart(fig, width="stretch")
            st.caption("Cada barra se interpreta frente al umbral de su propio modelo; "
                       "una puntuación más alta entre modelos no implica necesariamente mayor certeza.")
            if len({item.predicted_class for item in results}) > 1:
                st.info("Los modelos discrepan en la clase de este corte. Compara sus umbrales y "
                        "métricas de prueba antes de interpretar la diferencia.")


def render_performance(models) -> None:
    header("Rendimiento", "Métricas declaradas del conjunto de prueba")
    show = st.toggle("Mostrar métricas del conjunto de prueba", value=False)
    if not show:
        st.info("Activa el control para cargar y mostrar las métricas configuradas.")
        return
    records, matrices, efficiency, messages = load_all_metrics(models)
    for message in messages:
        st.info(message)
    if records.empty:
        st.warning("No hay métricas de prueba disponibles. Agrégalas y configura `metrics_path` por modelo.")
        return
    available = records["Modelo"].unique().tolist()
    selected = st.multiselect("Modelos para comparar", available, default=available)
    if not selected:
        st.info("Selecciona al menos un modelo.")
        return
    visible = records.loc[records["Modelo"].isin(selected)]
    st.plotly_chart(metric_chart(visible), width="stretch")
    st.dataframe(visible, width="stretch", hide_index=True)
    st.caption("Sensibilidad: proporción de cortes positivos detectados. Especificidad: "
               "proporción de cortes negativos reconocidos. Se calcularon sobre datos de prueba, "
               "con el umbral de cribado de cada modelo.")
    if not efficiency.empty:
        visible_efficiency = efficiency.loc[efficiency["Modelo"].isin(selected)]
        st.subheader("Costo de los modelos")
        st.dataframe(visible_efficiency, width="stretch", hide_index=True)
        if "ms por imagen" in visible_efficiency:
            fig = px.bar(visible_efficiency, x="Modelo", y="ms por imagen", color="Modelo",
                         color_discrete_sequence=MODEL_COLORS,
                         title="Tiempo de inferencia medido en el conjunto de prueba")
            st.plotly_chart(fig, width="stretch")
            st.caption("Menos milisegundos implica respuesta más rápida en el entorno de prueba; "
                       "la latencia de tu equipo puede ser diferente.")
            auc = visible.loc[visible["Métrica"].eq("ROC-AUC"), ["Modelo", "Valor"]]
            tradeoff = visible_efficiency.merge(auc, on="Modelo", how="inner")
            if not tradeoff.empty:
                fig = px.scatter(tradeoff, x="ms por imagen", y="Valor", color="Modelo",
                                 text="Modelo", color_discrete_sequence=MODEL_COLORS,
                                 labels={"Valor": "ROC-AUC en prueba"},
                                 title="Discriminación frente a tiempo de inferencia")
                fig.update_traces(textposition="top center", marker_size=14)
                st.plotly_chart(fig, width="stretch")
                st.caption("Más arriba indica mejor discriminación en prueba; más a la izquierda, "
                           "menor tiempo por corte en el entorno de medición.")
    for model_name, matrix in matrices.items():
        if model_name in selected:
            st.plotly_chart(confusion_figure(model_name, matrix), width="stretch")
            st.caption("Filas: clase real; columnas: clase predicha. Los falsos negativos "
                       "son fracturas en la fila inferior clasificadas como sin fractura.")


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
