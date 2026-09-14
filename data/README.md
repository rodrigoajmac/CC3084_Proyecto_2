# `data/` — cachés generados

Esta carpeta guarda tablas derivadas que **sí se versionan**: pesan poco y
ahorran horas de cómputo en Kaggle. Los datos originales (DICOM, NIfTI) nunca
se commitean.

| Archivo | Lo genera | Contenido |
|---|---|---|
| `meta_train.csv` | `notebooks/05-calidad.ipynb` vía `src/metadata.py` | 2 019 × 19 — metadatos DICOM con esquema completo (`z_dir`, `slice_gap`, `rows`, `cols`, `z_first`, `z_last`). |
| `reporte_calidad.csv` | `notebooks/05-calidad.ipynb` vía `src/quality.py` | Una fila por chequeo de calidad, con magnitud, severidad y acción. |
| `diccionario_variables.csv` | `notebooks/05-calidad.ipynb` vía `src/quality.py` | Tabla de variables con tipo y descripción (sección 4 del informe). |
| `volumen_segmentaciones.csv` | `notebooks/03-espacial.ipynb` | Conteo de voxeles por vértebra C1–C7 y T1–T12, para los 87 estudios segmentados. |

## Cómo regenerarlos

Los cuatro requieren el conjunto de la competencia montado, así que se producen
en Kaggle. Adjunta el dataset al notebook, ejecútalo completo y descarga los CSV
desde `/kaggle/working/data/` a esta carpeta.

`meta_train.csv` es la **ruta crítica**: `src/quality.py` depende de él para los
chequeos de orientación (`z_dir`), huecos de numeración (`slice_gap`) y tamaño de
imagen (`rows × cols`). Mientras no exista, esos chequeos se reportan como
`no_evaluable` en lugar de desaparecer del informe.
