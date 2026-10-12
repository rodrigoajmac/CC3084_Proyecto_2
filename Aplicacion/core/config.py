"""Lectura y validación estricta de configuración."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

APP_DIR = Path(__file__).resolve().parents[1]
MODEL_CONFIG_FILE = APP_DIR / "models.yaml"
APP_CONFIG_FILE = APP_DIR / "app_config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    path: Path
    framework: str
    format: str
    input_size: tuple[int, int]
    color_mode: str
    channels: int
    normalization: dict[str, Any]
    class_order: tuple[str, ...]
    fracture_index: int
    threshold: float
    output_type: str
    metrics_path: Path | None = None
    last_conv_layer: str | None = None

    @property
    def cache_key(self) -> str:
        stamp = self.path.stat().st_mtime_ns if self.path.exists() else 0
        return sha256(f"{self.path.resolve()}:{stamp}".encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["metrics_path"] = str(self.metrics_path) if self.metrics_path else None
        return result


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} debe contener un objeto YAML.")
    return value


def _resolve(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (APP_DIR / path).resolve()


def load_app_config() -> dict[str, Any]:
    return _yaml(APP_CONFIG_FILE)


def _parse_model(raw: dict[str, Any], index: int) -> ModelConfig:
    required = ["id", "name", "path", "framework", "format", "input_size", "color_mode",
                "channels", "normalization", "class_order", "fracture_index", "threshold", "output_type"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"faltan campos: {', '.join(missing)}")
    size = raw["input_size"]
    if not isinstance(size, list) or len(size) != 2 or any(int(v) <= 0 for v in size):
        raise ValueError("input_size debe ser [alto, ancho] con enteros positivos")
    classes = tuple(str(value) for value in raw["class_order"])
    fracture_index = int(raw["fracture_index"])
    if len(classes) != 2:
        raise ValueError("class_order debe contener exactamente dos clases")
    if fracture_index not in range(len(classes)):
        raise ValueError("fracture_index no corresponde a class_order")
    color = str(raw["color_mode"]).lower()
    channels = int(raw["channels"])
    if (color, channels) not in {("grayscale", 1), ("rgb", 3)}:
        raise ValueError("usa color_mode grayscale/channels 1 o rgb/channels 3")
    threshold = float(raw["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1")
    output_type = str(raw["output_type"]).lower()
    if output_type not in {"probability", "logit"}:
        raise ValueError("output_type debe ser probability o logit")
    framework = str(raw["framework"]).lower()
    model_format = str(raw["format"]).lower()
    if framework not in {"tensorflow", "pytorch"}:
        raise ValueError("framework debe ser tensorflow o pytorch")
    if framework == "pytorch" and model_format != "torchscript":
        raise ValueError("PyTorch solo admite format: torchscript; un state_dict requiere la arquitectura")
    if framework == "tensorflow" and model_format not in {"keras", "h5", "savedmodel"}:
        raise ValueError("formato TensorFlow no admitido")
    normalization = raw["normalization"]
    if not isinstance(normalization, dict) or "type" not in normalization:
        raise ValueError("normalization debe declarar type")
    norm_type = str(normalization["type"]).lower()
    if norm_type not in {"scale", "mean_std", "none"}:
        raise ValueError("normalization.type debe ser scale, mean_std o none")
    if norm_type == "scale" and "divisor" not in normalization:
        raise ValueError("normalization.scale requiere divisor")
    if norm_type == "mean_std":
        absent = [key for key in ("divisor", "mean", "std") if key not in normalization]
        if absent:
            raise ValueError(f"normalization.mean_std requiere: {', '.join(absent)}")
    return ModelConfig(
        id=str(raw["id"]), name=str(raw["name"]), path=_resolve(str(raw["path"])),
        framework=framework, format=model_format, input_size=(int(size[0]), int(size[1])),
        color_mode=color, channels=channels, normalization=normalization,
        class_order=classes, fracture_index=fracture_index, threshold=threshold,
        output_type=output_type, metrics_path=_resolve(raw.get("metrics_path")),
        last_conv_layer=raw.get("last_conv_layer"),
    )


def load_model_configs() -> tuple[list[ModelConfig], list[str]]:
    document = _yaml(MODEL_CONFIG_FILE)
    entries = document.get("models", [])
    if not isinstance(entries, list):
        return [], ["`models` debe ser una lista."]
    models: list[ModelConfig] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        try:
            if not isinstance(raw, dict):
                raise ValueError("la entrada debe ser un objeto")
            model = _parse_model(raw, index)
            if model.id in seen:
                raise ValueError(f"id duplicado: {model.id}")
            seen.add(model.id)
            models.append(model)
        except Exception as exc:
            errors.append(f"Modelo #{index + 1}: {exc}")
    return models, errors
