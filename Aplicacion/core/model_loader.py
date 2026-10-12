"""Carga perezosa y cacheada de modelos externos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from core.config import ModelConfig


@dataclass(frozen=True)
class Status:
    ready: bool
    message: str


@dataclass
class RuntimeModel:
    framework: str
    model: Any
    device: str = "cpu"


def model_status(config: ModelConfig) -> Status:
    if not config.path.exists():
        return Status(False, f"archivo no encontrado: {config.path}")
    if config.framework == "tensorflow" and config.format in {"keras", "h5"}:
        expected = ".keras" if config.format == "keras" else ".h5"
        if config.path.suffix.lower() != expected:
            return Status(False, f"se esperaba un archivo {expected}")
    if config.format == "savedmodel" and not config.path.is_dir():
        return Status(False, "SavedModel debe ser un directorio")
    if config.format == "torchscript" and config.path.suffix.lower() not in {".pt", ".pth"}:
        return Status(False, "TorchScript debe usar extensión .pt o .pth")
    return Status(True, "listo para cargar")


@st.cache_resource(show_spinner=False)
def load_model(cache_key: str, raw: dict[str, Any]) -> RuntimeModel:
    """Carga una vez por archivo/fecha de modificación. cache_key invalida la caché."""
    del cache_key
    path = Path(raw["path"])
    if raw["framework"] == "tensorflow":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow no está instalado. Instala `tensorflow`.") from exc
        if raw["format"] == "savedmodel":
            try:
                model = tf.keras.layers.TFSMLayer(str(path), call_endpoint="serving_default")
            except Exception as exc:
                raise RuntimeError(f"No se pudo abrir SavedModel: {exc}") from exc
        else:
            model = tf.keras.models.load_model(path, compile=False)
        return RuntimeModel("tensorflow", model)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch no está instalado. Instala `torch`.") from exc
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"El archivo no es un TorchScript válido: {exc}") from exc
    return RuntimeModel("pytorch", model)
