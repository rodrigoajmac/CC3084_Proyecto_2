"""
src/metadata.py
Extracción rápida y segura de metadatos DICOM desde cabeceras (3 cortes/estudio).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import pydicom
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from src.config import TRAIN_IMAGES_DIR, META_CACHE_FILE


def obtener_cortes_ordenados(study_dir: Path) -> List[int]:
    """Retorna los números de corte de un estudio ordenados numéricamente."""
    return sorted(
        int(f.stem)
        for f in study_dir.iterdir()
        if f.suffix.lower() == ".dcm" and f.stem.isdigit()
    )


def meta_estudio(uid: str, images_base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Extrae metadatos volumétricos y técnicos leyendo únicamente 3 cortes

    (primero, intermedio y último) para minimizar I/O y memoria.
    """
    base = images_base_dir or TRAIN_IMAGES_DIR
    study_dir = base / uid

    if not study_dir.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {study_dir}")

    slice_nums = obtener_cortes_ordenados(study_dir)
    n_slices = len(slice_nums)
    if n_slices == 0:
        raise ValueError(f"Estudio {uid} no contiene archivos DICOM válidos.")

    # Identificar índices de corte: primero, medio, último
    idx_first = slice_nums[0]
    idx_mid = slice_nums[n_slices // 2]
    idx_last = slice_nums[-1]

    # Verificar huecos en la numeración consecutiva
    slice_gap = int(idx_last - idx_first + 1 != n_slices)

    # Leer únicamente cabeceras (stop_before_pixels=True)
    dcm_first = pydicom.dcmread(study_dir / f"{idx_first}.dcm", stop_before_pixels=True)
    dcm_mid = pydicom.dcmread(study_dir / f"{idx_mid}.dcm", stop_before_pixels=True)
    dcm_last = pydicom.dcmread(study_dir / f"{idx_last}.dcm", stop_before_pixels=True)

    # Dimensiones y espaciado
    rows = getattr(dcm_mid, "Rows", np.nan)
    cols = getattr(dcm_mid, "Columns", np.nan)
    slice_thickness = float(getattr(dcm_mid, "SliceThickness", np.nan))

    pixel_spacing = getattr(dcm_mid, "PixelSpacing", [np.nan, np.nan])
    pixel_spacing_x = float(pixel_spacing[0]) if len(pixel_spacing) > 0 else np.nan
    pixel_spacing_y = float(pixel_spacing[1]) if len(pixel_spacing) > 1 else np.nan

    # Coordenadas anatómicas Z (ImagePositionPatient[2])
    ipp_first = getattr(dcm_first, "ImagePositionPatient", None)
    ipp_last = getattr(dcm_last, "ImagePositionPatient", None)

    z_first = float(ipp_first[2]) if ipp_first is not None and len(ipp_first) >= 3 else np.nan
    z_last = float(ipp_last[2]) if ipp_last is not None and len(ipp_last) >= 3 else np.nan

    if not np.isnan(z_first) and not np.isnan(z_last):
        z_extent = float(abs(z_last - z_first))
        # z_dir: craneocaudal vs caudocraneal
        z_dir = "ascendente" if z_last > z_first else "descendente"
    else:
        z_extent = np.nan
        z_dir = "desconocido"

    # Parámetros técnicos del escáner (atributos opcionales)
    kvp = float(getattr(dcm_mid, "KVP", np.nan))
    manufacturer = str(getattr(dcm_mid, "Manufacturer", "Desconocido")).strip()
    model = str(getattr(dcm_mid, "ManufacturerModelName", "Desconocido")).strip()
    kernel = str(getattr(dcm_mid, "ConvolutionKernel", "Desconocido")).strip()
    patient_sex = str(getattr(dcm_mid, "PatientSex", "Desconocido")).strip()
    patient_age = str(getattr(dcm_mid, "PatientAge", "Desconocido")).strip()
    study_desc = str(getattr(dcm_mid, "StudyDescription", "Desconocido")).strip()

    return {
        "StudyInstanceUID": uid,
        "n_slices": n_slices,
        "slice_gap": slice_gap,
        "slice_thickness": slice_thickness,
        "pixel_spacing_x": pixel_spacing_x,
        "pixel_spacing_y": pixel_spacing_y,
        "rows": rows,
        "cols": cols,
        "z_first": z_first,
        "z_last": z_last,
        "z_extent": z_extent,
        "z_dir": z_dir,
        "kvp": kvp,
        "manufacturer": manufacturer,
        "model": model,
        "kernel": kernel,
        "patient_sex": patient_sex,
        "patient_age": patient_age,
        "study_desc": study_desc,
    }


def construir_tabla(
    uids: List[str],
    cache_path: Path = META_CACHE_FILE,
    force_recompute: bool = False
) -> pd.DataFrame:
    """Extrae metadatos para la lista de UIDs con caché local en CSV

    y tolerancia a fallos por estudio.
    """
    if not force_recompute and cache_path.exists():
        print(f"[CACHE] Cargando metadatos precalculados desde {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[EXTRACCION] Procesando {len(uids)} estudios tomográficos...")
    registros = []
    errores = []

    for uid in tqdm(uids, desc="Extrayendo cabeceras DICOM"):
        try:
            reg = meta_estudio(uid)
            registros.append(reg)
        except Exception as exc:
            errores.append({"StudyInstanceUID": uid, "error": str(exc)})

    if errores:
        print(f"[ALERTA] Se presentaron {len(errores)} fallos durante la extracción.")

    df_meta = pd.DataFrame(registros)
    df_meta.to_csv(cache_path, index=False)
    print(f"[CACHE] Metadatos exportados exitosamente a {cache_path}")
    return df_meta