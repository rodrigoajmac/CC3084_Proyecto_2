"""Lectura y visualización de métricas existentes, sin recalcularlas."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px

from core.config import ModelConfig

DISPLAY = {
    "sensitivity": "Sensibilidad", "specificity": "Especificidad",
    "precision": "Precisión", "f1": "F1", "f1_score": "F1", "roc_auc": "ROC-AUC",
    "accuracy": "Exactitud",
}

MODEL_COLORS = ["#145A73", "#BD6430", "#5B477F"]


def _read(config: ModelConfig) -> tuple[list[dict], list[list[float]] | None, dict]:
    path = config.metrics_path
    if path is None or not path.is_file():
        return [], None, {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("el JSON debe ser un objeto")
        matrix = data.get("confusion_matrix")
        records = []
        for key, display in DISPLAY.items():
            if key in data and isinstance(data[key], (int, float)):
                records.append({"Modelo": config.name, "Métrica": display, "Valor": float(data[key])})
        efficiency = {"Modelo": config.name}
        for key, label in (("ms_por_imagen", "ms por imagen"),
                           ("segundos_entrenamiento", "Segundos de entrenamiento"),
                           ("parametros", "Parámetros"), ("n_test", "Cortes de prueba")):
            if isinstance(data.get(key), (int, float)):
                efficiency[label] = float(data[key])
        return records, matrix, efficiency
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        lower = {str(column).lower(): column for column in frame.columns}
        records = []
        if {"metric", "value"}.issubset(lower):
            for _, row in frame.iterrows():
                key = str(row[lower["metric"]]).lower()
                if key in DISPLAY:
                    records.append({"Modelo": config.name, "Métrica": DISPLAY[key],
                                    "Valor": float(row[lower["value"]])})
        elif len(frame) == 1:
            for key, display in DISPLAY.items():
                if key in lower and pd.notna(frame.iloc[0][lower[key]]):
                    records.append({"Modelo": config.name, "Métrica": display,
                                    "Valor": float(frame.iloc[0][lower[key]])})
        return records, None, {}
    raise ValueError("solo se admiten métricas JSON o CSV")


def load_all_metrics(models: list[ModelConfig]):
    rows: list[dict] = []
    efficiency_rows: list[dict] = []
    matrices: dict[str, list[list[float]]] = {}
    messages: list[str] = []
    for config in models:
        if config.metrics_path is None:
            messages.append(f"{config.name}: no tiene metrics_path configurado.")
            continue
        if not config.metrics_path.is_file():
            messages.append(f"{config.name}: no se encontró {config.metrics_path}.")
            continue
        try:
            records, matrix, efficiency = _read(config)
            rows.extend(records)
            if efficiency:
                efficiency_rows.append(efficiency)
            if matrix is not None:
                value = np.asarray(matrix)
                if value.shape == (2, 2) and np.isfinite(value.astype(float)).all():
                    matrices[config.name] = matrix
                else:
                    messages.append(f"{config.name}: confusion_matrix debe ser numérica de 2 × 2.")
            if not records and matrix is None:
                messages.append(f"{config.name}: el archivo no contiene métricas reconocidas.")
        except Exception as exc:
            messages.append(f"{config.name}: no se pudieron leer las métricas ({exc}).")
    return (pd.DataFrame(rows, columns=["Modelo", "Métrica", "Valor"]), matrices,
            pd.DataFrame(efficiency_rows), messages)


def metric_chart(frame: pd.DataFrame):
    bounded = frame.copy()
    chart_range = [0, 1] if bounded["Valor"].between(0, 1).all() else None
    return px.bar(bounded, x="Métrica", y="Valor", color="Modelo", barmode="group",
                  color_discrete_sequence=MODEL_COLORS, range_y=chart_range,
                  title="Comparación de métricas de prueba")


def confusion_figure(model_name: str, matrix):
    return px.imshow(np.asarray(matrix, dtype=float), text_auto=True,
                     x=["Pred. sin fractura", "Pred. fractura"],
                     y=["Real sin fractura", "Real fractura"],
                     color_continuous_scale="Blues", title=f"Matriz de confusión — {model_name}")
