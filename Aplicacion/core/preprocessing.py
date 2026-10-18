"""Preparación de cortes CT con el mismo flujo de notebooks/06-modelado.ipynb."""

from __future__ import annotations

import numpy as np
from PIL import Image
from io import BytesIO

from core.config import ModelConfig


def prepare_dicom(data: bytes, max_pixels: int) -> Image.Image:
    """Convierte un único corte DICOM a los píxeles uint8 usados al entrenar."""
    import pydicom

    try:
        dataset = pydicom.dcmread(BytesIO(data))
        rows, cols = int(dataset.Rows), int(dataset.Columns)
        if rows < 16 or cols < 16 or rows * cols > max_pixels:
            raise ValueError("Dimensiones DICOM fuera de los límites admitidos")
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise ValueError("Carga un solo corte DICOM, no un estudio de varios fotogramas")
        if getattr(dataset, "Modality", "CT") != "CT":
            raise ValueError("Los modelos se entrenaron con tomografías CT")
        pixels = dataset.pixel_array.astype(np.float32)
    except (AttributeError, TypeError, pydicom.errors.InvalidDicomError) as exc:
        raise ValueError(f"No se pudo leer el corte DICOM: {exc}") from exc
    if pixels.ndim != 2 or pixels.shape != (rows, cols):
        raise ValueError("Se espera un corte CT monocromático de dos dimensiones")
    try:
        slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
        intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("RescaleSlope o RescaleIntercept no es numérico") from exc
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("Los parámetros de conversión a HU deben ser finitos")

    hu = pixels * slope + intercept
    bone_window = np.clip((hu - (400.0 - 1800.0 / 2)) / 1800.0, 0, 1)
    # Misma interpolación y cuantización que la celda carga_corte del notebook.
    import tensorflow as tf

    resized = tf.image.resize(bone_window[..., None], (224, 224), method="bilinear").numpy()
    image = np.clip(resized * 255, 0, 255).astype(np.uint8)[..., 0]
    return Image.fromarray(image, mode="L")


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
