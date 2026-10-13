"""
src/quality.py
Chequeos de calidad de datos del Proyecto 2 (Miembro D).

La función principal es `reporte_calidad()`. Devuelve un DataFrame con **una
fila por problema detectado** y su magnitud cuantificada, que es el insumo de la
sección "Descripción de los datos" del informe.

Principio de diseño: este módulo *no corrige* nada. El dataset RSNA contiene
variabilidad legítima (12 instituciones, 9 países) y ambigüedad diagnóstica real.
Eliminar esos casos sesgaría el análisis. El reporte documenta la magnitud del
problema y propone una acción, pero la decisión queda escrita en el informe.

Segundo principio: tolerancia a esquemas parciales. La tabla de metadatos puede
provenir de `src/metadata.py` (esquema completo, con `z_dir`, `slice_gap`,
`rows`, `cols`) o de una extracción más ligera hecha en un notebook. Cuando una
columna necesaria no está, el chequeo se reporta como `no_evaluable` en lugar de
fallar: así queda constancia de que la verificación no se hizo, en vez de
desaparecer silenciosamente del informe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - depende del entorno
    from src.config import LEVELS
except Exception:  # pragma: no cover
    LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# Columnas del reporte, en el orden en que se leen en el informe.
COLUMNAS_REPORTE = [
    "id",
    "dimension",
    "hallazgo",
    "magnitud",
    "unidad",
    "porcentaje",
    "severidad",
    "estado",
    "accion",
]

# Severidades admitidas; se usan para colorear la figura del reporte.
SEVERIDADES = ("alta", "media", "baja", "informativa")


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _fila(
    id_: str,
    dimension: str,
    hallazgo: str,
    magnitud: Any,
    unidad: str,
    porcentaje: float | None,
    severidad: str,
    accion: str,
    estado: str = "detectado",
) -> dict[str, Any]:
    """Construye una fila del reporte con el esquema fijo."""
    if severidad not in SEVERIDADES:
        raise ValueError(f"Severidad no admitida: {severidad!r}")
    return {
        "id": id_,
        "dimension": dimension,
        "hallazgo": hallazgo,
        "magnitud": magnitud,
        "unidad": unidad,
        "porcentaje": None if porcentaje is None else round(float(porcentaje), 2),
        "severidad": severidad,
        "estado": estado,
        "accion": accion,
    }


def _no_evaluable(id_: str, dimension: str, hallazgo: str, columnas: Sequence[str]) -> dict[str, Any]:
    """Fila para un chequeo que no se pudo ejecutar por falta de columnas."""
    faltantes = ", ".join(columnas)
    return _fila(
        id_=id_,
        dimension=dimension,
        hallazgo=f"{hallazgo} — no evaluable",
        magnitud=np.nan,
        unidad="—",
        porcentaje=None,
        severidad="informativa",
        estado="no_evaluable",
        accion=(f"Regenerar la tabla de metadatos con src/metadata.py: falta(n) "
                f"la(s) columna(s) {faltantes}."),
    )


def _pct(parte: float, total: float) -> float | None:
    """Porcentaje seguro ante denominador cero."""
    return None if not total else 100.0 * parte / total


def _uids(valor: Iterable[str] | pd.Series | pd.DataFrame | None,
          columna: str = "StudyInstanceUID") -> set[str]:
    """Normaliza distintas formas de pasar un conjunto de UIDs."""
    if valor is None:
        return set()
    if isinstance(valor, pd.DataFrame):
        return set(valor[columna].astype(str))
    if isinstance(valor, pd.Series):
        return set(valor.astype(str))
    return {str(v) for v in valor}


def _existe(df: pd.DataFrame | None, *columnas: str) -> bool:
    return df is not None and all(col in df.columns for col in columnas)


# ---------------------------------------------------------------------------
# Chequeos individuales
# ---------------------------------------------------------------------------
def _chequeos_etiquetas(train: pd.DataFrame) -> list[dict[str, Any]]:
    """Integridad interna de train.csv."""
    filas: list[dict[str, Any]] = []
    n = len(train)
    niveles = [c for c in LEVELS if c in train.columns]

    # 1. Duplicados de estudio
    duplicados = int(train["StudyInstanceUID"].duplicated().sum())
    filas.append(_fila(
        "ETI-01", "Etiquetas", "Estudios duplicados en train.csv",
        duplicados, "estudios", _pct(duplicados, n),
        "alta" if duplicados else "baja",
        "Cada StudyInstanceUID debe aparecer una sola vez; si no, colapsar filas antes de particionar.",
        estado="detectado" if duplicados else "sin_hallazgos",
    ))

    # 2. Nulos en las 8 etiquetas
    nulos = int(train[["patient_overall"] + niveles].isna().sum().sum())
    filas.append(_fila(
        "ETI-02", "Etiquetas", "Valores nulos en las 8 etiquetas",
        nulos, "celdas", _pct(nulos, n * (len(niveles) + 1)),
        "alta" if nulos else "baja",
        "Las etiquetas son binarias obligatorias; un nulo invalida la fila para entrenamiento supervisado.",
        estado="detectado" if nulos else "sin_hallazgos",
    ))

    # 3. Inconsistencia patient_overall vs OR(C1..C7).
    #    La documentación oficial advierte que puede haber positivos sin nivel
    #    asignado (incertidumbre diagnóstica real). Se cuantifican los dos sentidos.
    calculado = train[niveles].max(axis=1)
    pos_sin_nivel = int(((train["patient_overall"] == 1) & (calculado == 0)).sum())
    nivel_sin_pos = int(((train["patient_overall"] == 0) & (calculado == 1)).sum())
    total_inconsistente = pos_sin_nivel + nivel_sin_pos
    filas.append(_fila(
        "ETI-03", "Etiquetas", "patient_overall = 1 sin ningún nivel C1–C7 marcado",
        pos_sin_nivel, "estudios", _pct(pos_sin_nivel, n),
        "media" if pos_sin_nivel else "baja",
        ("No corregir: refleja incertidumbre diagnóstica del informe radiológico original. "
         "Documentar y decidir explícitamente si se excluyen de la tarea por nivel."),
        estado="detectado" if pos_sin_nivel else "sin_hallazgos",
    ))
    filas.append(_fila(
        "ETI-04", "Etiquetas", "Nivel C1–C7 marcado con patient_overall = 0",
        nivel_sin_pos, "estudios", _pct(nivel_sin_pos, n),
        "alta" if nivel_sin_pos else "baja",
        "Este sentido sí sería contradictorio; si aparece, escalar a revisión manual.",
        estado="detectado" if nivel_sin_pos else "sin_hallazgos",
    ))
    filas.append(_fila(
        "ETI-05", "Etiquetas", "Inconsistencia total patient_overall vs OR(C1–C7)",
        total_inconsistente, "estudios", _pct(total_inconsistente, n),
        "media" if total_inconsistente else "baja",
        "Resumen de ETI-03 y ETI-04.",
        estado="detectado" if total_inconsistente else "sin_hallazgos",
    ))

    # 4. Desbalance de la etiqueta principal
    positivos = int(train["patient_overall"].sum())
    filas.append(_fila(
        "ETI-06", "Etiquetas", "Prevalencia de patient_overall = 1",
        positivos, "estudios", _pct(positivos, n),
        "informativa",
        "Determina si el desbalance obliga a ponderar la pérdida; también es la variable de estratificación.",
        estado="informativo",
    ))

    # 5. Desbalance entre niveles: el nivel más raro condiciona la partición
    conteos = train[niveles].sum()
    nivel_min, nivel_max = conteos.idxmin(), conteos.idxmax()
    filas.append(_fila(
        "ETI-07", "Etiquetas", f"Nivel menos representado ({nivel_min}) frente al más frecuente ({nivel_max})",
        f"{int(conteos[nivel_min])} vs {int(conteos[nivel_max])}", "estudios",
        _pct(conteos[nivel_min], conteos[nivel_max]),
        "media",
        ("Razón entre el nivel más raro y el más frecuente. Un desbalance fuerte exige "
         "estratificar también por nivel, no solo por patient_overall."),
        estado="informativo",
    ))
    return filas


def _chequeos_integridad(train: pd.DataFrame, meta: pd.DataFrame | None,
                         images_dir: Path | str | None) -> list[dict[str, Any]]:
    """Correspondencia entre train.csv y las carpetas de imágenes."""
    filas: list[dict[str, Any]] = []
    n = len(train)
    uids_train = _uids(train)

    if meta is not None and "StudyInstanceUID" in meta.columns:
        uids_meta = _uids(meta)
        sin_meta = len(uids_train - uids_meta)
        sobrantes = len(uids_meta - uids_train)
        filas.append(_fila(
            "INT-01", "Integridad", "Estudios de train.csv sin fila en la tabla de metadatos",
            sin_meta, "estudios", _pct(sin_meta, n),
            "alta" if sin_meta else "baja",
            "Indica fallos de lectura DICOM o carpetas ausentes; revisar la lista de errores de construir_tabla().",
            estado="detectado" if sin_meta else "sin_hallazgos",
        ))
        filas.append(_fila(
            "INT-02", "Integridad", "Estudios con metadatos que no están en train.csv",
            sobrantes, "estudios", _pct(sobrantes, max(len(uids_meta), 1)),
            "media" if sobrantes else "baja",
            "Normalmente son carpetas de test mezcladas; excluirlas antes de cruzar con la etiqueta.",
            estado="detectado" if sobrantes else "sin_hallazgos",
        ))
    else:
        filas.append(_no_evaluable("INT-01", "Integridad",
                                   "Cruce train.csv vs tabla de metadatos",
                                   ["meta.StudyInstanceUID"]))

    if images_dir is not None:
        base = Path(images_dir)
        if base.is_dir():
            carpetas = {p.name for p in base.iterdir() if p.is_dir()}
            faltantes = len(uids_train - carpetas)
            filas.append(_fila(
                "INT-03", "Integridad", "Estudios de train.csv sin carpeta de imágenes",
                faltantes, "estudios", _pct(faltantes, n),
                "alta" if faltantes else "baja",
                "Un estudio sin píxeles no es utilizable; excluirlo y reportar la exclusión.",
                estado="detectado" if faltantes else "sin_hallazgos",
            ))
        else:
            filas.append(_no_evaluable("INT-03", "Integridad",
                                       "Estudios sin carpeta de imágenes",
                                       [str(base)]))
    return filas


def _chequeos_metadatos(meta: pd.DataFrame | None, n_train: int) -> list[dict[str, Any]]:
    """Homogeneidad y completitud de los metadatos DICOM."""
    filas: list[dict[str, Any]] = []
    if meta is None or meta.empty:
        return [_no_evaluable("MET-00", "Metadatos DICOM",
                              "Chequeos de metadatos", ["meta_train.csv"])]
    n = len(meta)

    # 1. Huecos en la numeración de cortes
    if "slice_gap" in meta.columns:
        con_hueco = int(pd.to_numeric(meta["slice_gap"], errors="coerce").fillna(0).gt(0).sum())
        filas.append(_fila(
            "MET-01", "Metadatos DICOM", "Estudios con huecos en la numeración de cortes",
            con_hueco, "estudios", _pct(con_hueco, n),
            "media" if con_hueco else "baja",
            ("El nombre del archivo es InstanceNumber; un hueco significa que la pila no es "
             "contigua. No apilar por nombre: ordenar por ImagePositionPatient[2]."),
            estado="detectado" if con_hueco else "sin_hallazgos",
        ))
    else:
        filas.append(_no_evaluable("MET-01", "Metadatos DICOM",
                                   "Huecos en la numeración de cortes", ["slice_gap"]))

    # 2. Dirección del eje z
    if "z_dir" in meta.columns:
        dir_norm = meta["z_dir"].astype(str).str.strip().str.lower()
        descendentes = int(dir_norm.eq("descendente").sum())
        desconocidos = int(dir_norm.isin(["desconocido", "nan", "none", ""]).sum())
        filas.append(_fila(
            "MET-02", "Metadatos DICOM", "Estudios con eje z descendente (orden craneocaudal invertido)",
            descendentes, "estudios", _pct(descendentes, n),
            "alta" if descendentes else "baja",
            ("Normalizar la orientación ordenando por ImagePositionPatient[2] antes de "
             "cualquier reconstrucción sagital, recorte o superposición de máscara."),
            estado="detectado" if descendentes else "sin_hallazgos",
        ))
        filas.append(_fila(
            "MET-03", "Metadatos DICOM", "Estudios sin ImagePositionPatient legible",
            desconocidos, "estudios", _pct(desconocidos, n),
            "alta" if desconocidos else "baja",
            "Sin posición z no se puede garantizar el orden anatómico; usar InstanceNumber solo como respaldo declarado.",
            estado="detectado" if desconocidos else "sin_hallazgos",
        ))
    else:
        filas.append(_no_evaluable("MET-02", "Metadatos DICOM",
                                   "Dirección del eje z", ["z_dir"]))

    # 3. Metadatos ausentes por columna (una fila por columna con faltantes)
    for col in meta.columns:
        if col == "StudyInstanceUID":
            continue
        serie = meta[col]
        ausentes = int(serie.isna().sum())
        if serie.dtype == object:  # los tags opcionales se rellenan con "Desconocido"
            ausentes += int(serie.astype(str).str.strip().str.lower()
                            .isin(["desconocido", "unknown", "", "none", "nan"]).sum())
        if ausentes:
            filas.append(_fila(
                f"MET-NA-{col}", "Metadatos DICOM", f"Metadato ausente o desconocido: {col}",
                ausentes, "estudios", _pct(ausentes, n),
                "alta" if ausentes == n else ("media" if ausentes > 0.2 * n else "baja"),
                ("Columna constante: no aporta información y debe excluirse del análisis."
                 if ausentes == n else
                 "Imputar no aplica a un tag DICOM ausente; tratar 'Desconocido' como categoría propia."),
                estado="detectado",
            ))

    # 4. Heterogeneidad de la resolución
    for col, id_, severidad, accion in [
        ("slice_thickness", "MET-04", "media",
         ("El criterio de inclusión oficial fue corte fino sin contraste. La dispersión "
          "observada es variabilidad de reconstrucción entre sitios: remuestrear a "
          "espaciado isotrópico 1x1x1 mm antes de comparar volúmenes.")),
        ("pixel_spacing", "MET-05", "media",
         "Distintos campos de visión: el mismo número de píxeles cubre distintos milímetros. Remuestrear."),
        ("pixel_spacing_x", "MET-05x", "media",
         "Distintos campos de visión: remuestrear a espaciado isotrópico."),
    ]:
        if col in meta.columns:
            valores = pd.to_numeric(meta[col], errors="coerce").dropna().round(4)
            distintos = int(valores.nunique())
            filas.append(_fila(
                id_, "Metadatos DICOM", f"Valores distintos de {col}",
                distintos, "valores únicos", None,
                severidad if distintos > 1 else "baja", accion,
                estado="detectado" if distintos > 1 else "sin_hallazgos",
            ))

    # 5. Combinaciones de tamaño de imagen
    if _existe(meta, "rows", "cols"):
        combos = (meta["rows"].astype("Int64").astype(str) + "x"
                  + meta["cols"].astype("Int64").astype(str))
        distintos = int(combos.nunique())
        fuera = int((~combos.eq("512x512")).sum())
        filas.append(_fila(
            "MET-06", "Metadatos DICOM", "Combinaciones distintas de rows x cols",
            distintos, "combinaciones", _pct(fuera, n),
            "media" if distintos > 1 else "baja",
            ("El porcentaje indica estudios cuyo corte no es 512x512. Si el análisis de "
             "área de lesión asume 512x512, hay que recalcularlo por estudio."),
            estado="detectado" if distintos > 1 else "sin_hallazgos",
        ))
    else:
        filas.append(_no_evaluable("MET-06", "Metadatos DICOM",
                                   "Combinaciones de rows x cols", ["rows", "cols"]))

    # 6. Cobertura anatómica
    if "z_extent" in meta.columns:
        z = pd.to_numeric(meta["z_extent"], errors="coerce").dropna()
        if len(z):
            q1, q3 = z.quantile(0.25), z.quantile(0.75)
            iqr = q3 - q1
            atipicos = int(((z < q1 - 1.5 * iqr) | (z > q3 + 1.5 * iqr)).sum())
            filas.append(_fila(
                "MET-07", "Metadatos DICOM",
                f"Cobertura z atípica por IQR (rango {z.min():.0f}–{z.max():.0f} mm)",
                atipicos, "estudios", _pct(atipicos, n),
                "media" if atipicos else "baja",
                ("No eliminar: son protocolos legítimos (desde estudio cervical focalizado "
                 "hasta politrauma de cuerpo completo). Recortar a C1–C7 con las segmentaciones "
                 "homogeneiza el campo de visión sin descartar estudios."),
                estado="detectado" if atipicos else "sin_hallazgos",
            ))

    # 7. Número de cortes
    if "n_slices" in meta.columns:
        s = pd.to_numeric(meta["n_slices"], errors="coerce").dropna()
        if len(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            atipicos = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            filas.append(_fila(
                "MET-08", "Metadatos DICOM",
                f"Número de cortes atípico por IQR (rango {int(s.min())}–{int(s.max())})",
                atipicos, "estudios", _pct(atipicos, n),
                "baja", ("Variabilidad de protocolo, no error. Condiciona la memoria del pipeline: "
                         "el estudio más largo define el peor caso de carga."),
                estado="detectado" if atipicos else "sin_hallazgos",
            ))
    return filas


def _chequeos_anotacion(train: pd.DataFrame, bbox: pd.DataFrame | None,
                        seg_uids: Iterable[str] | None,
                        meta: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Cobertura y consistencia de las anotaciones espaciales."""
    filas: list[dict[str, Any]] = []
    n = len(train)
    uids_positivos = _uids(train.loc[train["patient_overall"].eq(1)])
    n_pos = len(uids_positivos)

    uids_bbox = _uids(bbox) if bbox is not None else set()
    uids_seg = _uids(seg_uids)

    if bbox is not None:
        filas.append(_fila(
            "ANO-01", "Anotación espacial", "Estudios sin bounding box",
            n - len(uids_bbox), "estudios", _pct(n - len(uids_bbox), n),
            "alta",
            ("La ausencia de caja NO significa ausencia de fractura. La localización es una "
             "etiqueta débil y parcial: no puede usarse como negativo para un detector."),
        ))
        cubiertos_pos = len(uids_bbox & uids_positivos)
        filas.append(_fila(
            "ANO-02", "Anotación espacial", "Casos positivos con bounding box",
            cubiertos_pos, "estudios positivos", _pct(cubiertos_pos, n_pos),
            "informativa",
            "Techo de datos disponible para entrenar una etapa de localización supervisada.",
            estado="informativo",
        ))
        # Cajas asignadas a estudios marcados como negativos: sería contradictorio
        bbox_en_negativos = len(uids_bbox - uids_positivos)
        filas.append(_fila(
            "ANO-03", "Anotación espacial", "Estudios con caja pero patient_overall = 0",
            bbox_en_negativos, "estudios", _pct(bbox_en_negativos, max(len(uids_bbox), 1)),
            "alta" if bbox_en_negativos else "baja",
            "Contradicción directa entre la etiqueta y la localización; escalar a revisión si aparece.",
            estado="detectado" if bbox_en_negativos else "sin_hallazgos",
        ))

        # Cajas que se salen del marco de la imagen
        if _existe(bbox, "x", "y", "width", "height"):
            if _existe(meta, "rows", "cols"):
                dims = meta[["StudyInstanceUID", "rows", "cols"]].copy()
                dims["StudyInstanceUID"] = dims["StudyInstanceUID"].astype(str)
                b = bbox.copy()
                b["StudyInstanceUID"] = b["StudyInstanceUID"].astype(str)
                b = b.merge(dims, on="StudyInstanceUID", how="left")
                alto = pd.to_numeric(b["rows"], errors="coerce").fillna(512)
                ancho = pd.to_numeric(b["cols"], errors="coerce").fillna(512)
            else:
                b = bbox
                alto = pd.Series(512.0, index=b.index)
                ancho = pd.Series(512.0, index=b.index)
            fuera = int((
                (b["x"] < 0) | (b["y"] < 0)
                | (b["x"] + b["width"] > ancho)
                | (b["y"] + b["height"] > alto)
            ).sum())
            filas.append(_fila(
                "ANO-04", "Anotación espacial", "Cajas que exceden el marco del corte",
                fuera, "cajas", _pct(fuera, len(b)),
                "media" if fuera else "baja",
                "Recortar la caja al marco antes de usarla como región de interés.",
                estado="detectado" if fuera else "sin_hallazgos",
            ))

        # Relación cajas/estudio: el número de cajas no es el número de fracturas
        if "slice_number" in bbox.columns:
            por_estudio = bbox.groupby("StudyInstanceUID").size()
            filas.append(_fila(
                "ANO-05", "Anotación espacial",
                f"Cajas por estudio anotado (mediana {por_estudio.median():.0f}, máx {por_estudio.max():.0f})",
                int(len(bbox)), "cajas", None,
                "informativa",
                ("Una fractura se anota en varios cortes consecutivos: contar cajas sobreestima "
                 "el número de lesiones independientes."),
                estado="informativo",
            ))

    if uids_seg:
        filas.append(_fila(
            "ANO-06", "Anotación espacial", "Estudios sin segmentación vertebral",
            n - len(uids_seg), "estudios", _pct(n - len(uids_seg), n),
            "alta",
            ("Solo una fracción mínima permite recortar C1–C7 con verdad de referencia. "
             "Para el resto habría que propagar con un modelo de segmentación, lo que "
             "introduce error no cuantificado en este EDA."),
        ))
        ambas = len(uids_seg & uids_bbox)
        filas.append(_fila(
            "ANO-07", "Anotación espacial", "Estudios con caja y segmentación simultáneas",
            ambas, "estudios", _pct(ambas, n),
            "media",
            ("Es el único subconjunto donde se puede validar que la máscara NIfTI quedó "
             "alineada con el DICOM (mirar si la caja cae dentro de la vértebra esperada)."),
            estado="informativo",
        ))

    return filas


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def reporte_calidad(
    train: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    bbox: pd.DataFrame | None = None,
    seg_uids: Iterable[str] | None = None,
    images_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Construye el reporte de calidad de datos del proyecto.

    Parameters
    ----------
    train : DataFrame de train.csv (StudyInstanceUID, patient_overall, C1..C7).
    meta : tabla de metadatos DICOM (salida de src.metadata.construir_tabla).
        Puede ser None o traer un subconjunto de columnas; los chequeos que
        dependan de columnas ausentes se marcan como `no_evaluable`.
    bbox : DataFrame de train_bounding_boxes.csv.
    seg_uids : iterable con los UID que tienen archivo .nii de segmentación.
    images_dir : ruta a train_images/ para verificar carpetas faltantes.

    Returns
    -------
    DataFrame con una fila por problema detectado y las columnas
    `id, dimension, hallazgo, magnitud, unidad, porcentaje, severidad, estado, accion`.
    """
    if "StudyInstanceUID" not in train.columns or "patient_overall" not in train.columns:
        raise ValueError("train debe contener StudyInstanceUID y patient_overall")

    filas: list[dict[str, Any]] = []
    filas += _chequeos_etiquetas(train)
    filas += _chequeos_integridad(train, meta, images_dir)
    filas += _chequeos_metadatos(meta, len(train))
    filas += _chequeos_anotacion(train, bbox, seg_uids, meta)

    reporte = pd.DataFrame(filas, columns=COLUMNAS_REPORTE)
    orden = {"alta": 0, "media": 1, "baja": 2, "informativa": 3}
    reporte = (reporte
               .assign(_orden=reporte["severidad"].map(orden))
               .sort_values(["_orden", "id"])
               .drop(columns="_orden")
               .reset_index(drop=True))
    return reporte


def resumen_reporte(reporte: pd.DataFrame) -> pd.DataFrame:
    """Conteo de hallazgos por severidad y estado, para el resumen del informe."""
    return (reporte
            .pivot_table(index="severidad", columns="estado", values="id",
                         aggfunc="count", fill_value=0)
            .reindex([s for s in SEVERIDADES if s in set(reporte["severidad"])])
            .astype(int))


def problemas_detectados(reporte: pd.DataFrame) -> pd.DataFrame:
    """Solo las filas con un problema real (excluye informativos y sin hallazgos)."""
    return reporte[reporte["estado"].eq("detectado")].reset_index(drop=True)


def tabla_variables(train: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Diccionario de datos: una fila por variable, con tipo y descripción.

    Alimenta directamente la tabla de la sección "Descripción de los datos".
    """
    descripciones = {
        "StudyInstanceUID": "Identificador único del estudio de TC; nombre de la carpeta en train_images/.",
        "patient_overall": "1 si el estudio presenta fractura cervical de cualquier nivel.",
        **{nivel: f"1 si la vértebra {nivel} presenta fractura." for nivel in LEVELS},
        "n_slices": "Número de cortes axiales (archivos DICOM) del estudio.",
        "slice_gap": "1 si la numeración de cortes no es contigua (hay huecos en InstanceNumber).",
        "slice_thickness": "Grosor físico del corte en mm (tag SliceThickness).",
        "pixel_spacing": "Distancia física entre píxeles contiguos en mm.",
        "pixel_spacing_x": "Distancia física entre píxeles en el eje x, en mm.",
        "pixel_spacing_y": "Distancia física entre píxeles en el eje y, en mm.",
        "rows": "Alto del corte en píxeles (tag Rows).",
        "cols": "Ancho del corte en píxeles (tag Columns).",
        "z_first": "Coordenada z del primer corte, en mm (ImagePositionPatient[2]).",
        "z_last": "Coordenada z del último corte, en mm.",
        "z_extent": "Longitud anatómica cubierta por el estudio en el eje craneocaudal, en mm.",
        "z_dir": "Sentido de apilamiento: ascendente o descendente.",
        "kvp": "Voltaje del tubo de rayos X en kilovoltios pico.",
        "manufacturer": "Fabricante del tomógrafo (tag Manufacturer).",
        "model": "Modelo del tomógrafo (tag ManufacturerModelName).",
        "kernel": "Kernel de reconstrucción usado (tag ConvolutionKernel).",
        "patient_sex": "Sexo del paciente (tag PatientSex).",
        "patient_age": "Edad del paciente (tag PatientAge).",
        "study_desc": "Descripción textual del estudio (tag StudyDescription).",
        "x": "Coordenada x de la esquina superior izquierda de la caja, en píxeles.",
        "y": "Coordenada y de la esquina superior izquierda de la caja, en píxeles.",
        "width": "Ancho de la caja en píxeles.",
        "height": "Alto de la caja en píxeles.",
        "slice_number": "InstanceNumber del corte anotado; concatenado con '.dcm' da el archivo.",
    }

    filas = []
    for origen, df in [("train.csv", train), ("meta_train.csv", meta)]:
        if df is None:
            continue
        for col in df.columns:
            serie = df[col]
            filas.append({
                "variable": col,
                "origen": origen,
                "tipo": str(serie.dtype),
                "valores_unicos": int(serie.nunique(dropna=True)),
                "nulos": int(serie.isna().sum()),
                "descripcion": descripciones.get(col, "Sin descripción registrada."),
            })
    return pd.DataFrame(filas).drop_duplicates(subset=["variable", "origen"]).reset_index(drop=True)
