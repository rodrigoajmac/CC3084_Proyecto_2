"""Preprocesamiento impulsado únicamente por configuración explícita."""

from __future__ import annotations

import numpy as np
from PIL import Image

from core.config import ModelConfig


def preprocess(image: Image.Image, config: ModelConfig) -> np.ndarray:
    mode = "L" if config.color_mode == "grayscale" else "RGB"
    converted = image.convert(mode)
    height, width = config.input_size
    resized = converted.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if config.channels == 1:
        array = array[..., None]
    norm = config.normalization
    kind = str(norm.get("type", "")).lower()
    if kind == "scale":
        if "divisor" not in norm:
            raise ValueError("normalization.scale requiere divisor")
        divisor = float(norm["divisor"])
        if divisor == 0:
            raise ValueError("El divisor de normalización no puede ser cero")
        array = array / divisor
    elif kind == "mean_std":
        if "divisor" not in norm or "mean" not in norm or "std" not in norm:
            raise ValueError("normalization.mean_std requiere divisor, mean y std")
        divisor = float(norm["divisor"])
        if divisor == 0:
            raise ValueError("El divisor de normalización no puede ser cero")
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        if mean.size not in {1, config.channels} or std.size not in {1, config.channels}:
            raise ValueError("mean/std deben tener uno o channels valores")
        if np.any(std == 0):
            raise ValueError("std no puede contener cero")
        array = (array / divisor - mean) / std
    elif kind == "none":
        pass
    else:
        raise ValueError(f"Tipo de normalización no admitido: {kind!r}")
    if not np.isfinite(array).all():
        raise ValueError("El preprocesamiento produjo valores no finitos")
    return array[None, ...]
