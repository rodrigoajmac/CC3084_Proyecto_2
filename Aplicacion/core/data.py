"""Descubrimiento y exploración segura de manifiestos tabulares."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.config import APP_DIR


def find_manifest(app_config: dict[str, Any]) -> Path | None:
    raw = app_config.get("manifest_path")
    if not raw:
        return None
    path = Path(raw)
    path = path if path.is_absolute() else (APP_DIR / path).resolve()
    return path if path.is_file() else None


def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError("El manifiesto debe ser CSV")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("El manifiesto supera el límite de 100 MB")
    return pd.read_csv(path)


def summarize_manifest(frame: pd.DataFrame):
    for label in ("patient_overall", "label", "class", "target"):
        if label in frame.columns:
            return label, frame[label].value_counts(dropna=False).sort_index()
    return None
