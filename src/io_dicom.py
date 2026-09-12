"""Utilidades de lectura y visualización para los volúmenes RSNA 2022.

Las funciones de este módulo evitan asumir que el nombre de un archivo DICOM
representa su posición anatómica. Los cortes se ordenan siempre por la
coordenada z de ImagePositionPatient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom

from src.config import SEGMENTATIONS_DIR, TRAIN_IMAGES_DIR


def _z_position(ds: pydicom.dataset.Dataset, fallback: float = 0.0) -> float:
    """Obtiene la coordenada z de un encabezado DICOM de forma defensiva."""
    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        try:
            return float(position[2])
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _dicom_paths(study_dir: Path) -> list[Path]:
    """Lista los DICOM de un estudio y valida que haya al menos uno."""
    paths = [path for path in study_dir.glob("*.dcm") if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No se encontraron cortes DICOM en {study_dir}")
    return paths


def cortes_ordenados(uid: str, images_dir: Path | str | None = None) -> list[Path]:
    """Devuelve las rutas de los cortes ordenadas por posición anatómica z.

    Si un corte carece de ImagePositionPatient, se usa InstanceNumber
    únicamente como respaldo. Solo se leen encabezados en esta primera pasada.
    """
    base = Path(images_dir) if images_dir is not None else TRAIN_IMAGES_DIR
    study_dir = base / str(uid)
    if not study_dir.is_dir():
        raise FileNotFoundError(f"No existe el estudio DICOM: {study_dir}")

    indexed: list[tuple[float, int, Path]] = []
    for path in _dicom_paths(study_dir):
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient", "InstanceNumber"],
        )
        try:
            instance = int(getattr(ds, "InstanceNumber", path.stem))
        except (TypeError, ValueError):
            instance = 0
        indexed.append((_z_position(ds, fallback=instance), instance, path))

    indexed.sort(key=lambda item: (item[0], item[1]))
    return [path for _, _, path in indexed]


def cargar_hu(path: Path | str) -> np.ndarray:
    """Lee un corte DICOM y convierte sus píxeles crudos a unidades Hounsfield."""
    ds = pydicom.dcmread(Path(path))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return pixels * slope + intercept


def ventana(hu: np.ndarray, centro: float, ancho: float) -> np.ndarray:
    """Aplica una ventana radiológica y devuelve valores float32 en [0, 1]."""
    if ancho <= 0:
        raise ValueError("El ancho de ventana debe ser mayor que cero")
    lower = float(centro) - float(ancho) / 2.0
    scaled = (np.asarray(hu, dtype=np.float32) - lower) / float(ancho)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def cargar_volumen(
    uid: str,
    paso: int = 1,
    images_dir: Path | str | None = None,
) -> np.ndarray:
    """Apila un estudio como volumen (z, y, x) ordenado por z real.

    paso permite submuestrear cortes para exploración rápida. No debe usarse
    para mediciones físicas sin ajustar el espaciado correspondiente.
    """
    if not isinstance(paso, int) or paso < 1:
        raise ValueError("paso debe ser un entero positivo")
    paths = cortes_ordenados(uid, images_dir=images_dir)[::paso]
    slices = [cargar_hu(path) for path in paths]
    shapes = {image.shape for image in slices}
    if len(shapes) != 1:
        raise ValueError(f"El estudio {uid} contiene cortes con dimensiones distintas: {shapes}")
    return np.stack(slices, axis=0)


def cargar_segmentacion(
    uid: str,
    segmentations_dir: Path | str | None = None,
) -> np.ndarray:
    """Carga una máscara NIfTI y la alinea a ejes DICOM (z, y, x).

    Aplica directamente sobre el arreglo original la corrección empírica documentada
    para este dataset, ajustada al orden z ascendente usado por cargar_volumen:
    transposición sagital a axial y volteo del eje vertical de la imagen. No se
    invierte z una segunda vez. Toda superposición debe validarse visualmente antes
    de utilizarla para mediciones anatómicas.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "cargar_segmentacion requiere nibabel; instálalo con 'pip install nibabel'"
        ) from exc

    base = Path(segmentations_dir) if segmentations_dir is not None else SEGMENTATIONS_DIR
    candidates: Iterable[Path] = (base / f"{uid}.nii", base / f"{uid}.nii.gz")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No existe segmentación para el estudio {uid} en {base}")

    nii = nib.load(path)
    segmentation = np.asanyarray(nii.dataobj)
    if segmentation.ndim != 3:
        raise ValueError(f"La segmentación {path.name} no es tridimensional")

    # NIfTI sagital (x, y, z) -> volumen axial (z, y, x), con corrección RSNA.
    segmentation = np.transpose(segmentation, (2, 1, 0))[:, ::-1, :]
    return np.rint(segmentation).astype(np.uint8, copy=False)
