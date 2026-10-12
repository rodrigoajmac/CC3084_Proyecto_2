"""Inferencia homogénea para modelos binarios o multiclase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from core.config import ModelConfig
from core.model_loader import RuntimeModel
from core.preprocessing import preprocess


class PredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    model: str
    predicted_class: str
    fracture_score: float
    score_label: str
    bounded_score: bool

    def as_row(self) -> dict[str, object]:
        return {"Modelo": self.model, "Clase predicha": self.predicted_class,
                "Puntuación de fractura": round(self.fracture_score, 5),
                "Tipo de puntuación": self.score_label}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80, 80)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _scores(output: object, config: ModelConfig) -> np.ndarray:
    if isinstance(output, dict):
        if len(output) != 1:
            raise PredictionError("El modelo devolvió varias salidas; configura un modelo con una salida única")
        output = next(iter(output.values()))
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise PredictionError("El modelo devolvió varias salidas no configuradas")
        output = output[0]
    if hasattr(output, "numpy"):
        output = output.numpy()
    raw = np.asarray(output, dtype=np.float64).squeeze()
    if raw.ndim > 1:
        raise PredictionError(f"Forma de salida no compatible: {raw.shape}")
    if raw.ndim == 0 or raw.size == 1:
        value = float(raw.reshape(-1)[0])
        fracture = value if config.output_type == "probability" else float(_sigmoid(np.array(value)))
        scores = np.empty(2, dtype=np.float64)
        scores[config.fracture_index] = fracture
        scores[1 - config.fracture_index] = 1.0 - fracture
        return scores
    if raw.size != len(config.class_order):
        raise PredictionError(
            f"La salida tiene {raw.size} valores y class_order tiene {len(config.class_order)} clases"
        )
    return raw if config.output_type == "probability" else _softmax(raw)


def predict(runtime: RuntimeModel, config: ModelConfig, image: Image.Image) -> Prediction:
    batch = preprocess(image, config)
    if runtime.framework == "tensorflow":
        output = runtime.model(batch, training=False)
    else:
        import torch
        tensor = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2))).float()
        with torch.inference_mode():
            output = runtime.model(tensor)
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
    scores = _scores(output, config)
    fracture_score = float(scores[config.fracture_index])
    if not np.isfinite(fracture_score):
        raise PredictionError("El modelo produjo una puntuación no finita")
    if config.output_type == "probability" and not 0 <= fracture_score <= 1:
        raise PredictionError("La salida declarada como probabilidad está fuera del intervalo [0, 1]")
    if len(config.class_order) == 2:
        fracture_class = config.class_order[config.fracture_index]
        other_class = config.class_order[1 - config.fracture_index]
        predicted = fracture_class if fracture_score >= config.threshold else other_class
    else:
        predicted = config.class_order[int(np.argmax(scores))]
    label = "Probabilidad del modelo (no clínica)" if config.output_type == "probability" else "Logit transformado"
    return Prediction(config.name, predicted, fracture_score, label, True)
