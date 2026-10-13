# SPECS — Proyecto 2: Análisis Exploratorio
## Reto 20 — RSNA 2022 Cervical Spine Fracture Detection

> Documento de especificaciones para Claude Code.
> **Léelo completo antes de escribir código.** Contiene decisiones técnicas ya tomadas
> y trampas conocidas del dataset que no son obvias.

---

## 1. Contexto

| | |
|---|---|
| Curso | CC3084 – Data Science, UVG, Semestre II 2026 |
| Entrega | **11 de septiembre de 2026** |
| Equipo | 4 integrantes (la nota es individual según contribuciones en Git) |
| Alcance | **Solo análisis exploratorio.** No se entrena ningún modelo en este proyecto. |
| Competencia | https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection |

### Entregables obligatorios
1. Informe en **PDF** con el análisis exploratorio.
2. **Repositorio GitHub** versionado (se usa para evaluar a cada miembro por separado).
3. **Presentación PowerPoint**.

### Aclaración sobre el enunciado
La guía del curso dice "radiografías". **Es incorrecto.** El dataset son tomografías
computarizadas (TC) axiales en DICOM: cada paciente es un volumen 3D de ~100–1000 cortes,
no una imagen 2D. Esto debe quedar explícito en el informe.

---

## 2. Dataset

### Estructura de archivos
```
<BASE>/
├── train.csv                 # StudyInstanceUID, patient_overall, C1..C7  (~2019 filas)
├── test.csv                  # StudyInstanceUID, prediction_type (test público mínimo)
├── train_bounding_boxes.csv  # StudyInstanceUID, x, y, width, height, slice_number
├── sample_submission.csv
├── segmentations/            # ~87 archivos <UID>.nii
└── train_images/<UID>/N.dcm  # ~700,000 archivos DICOM en total (~350 GB)
```

### Cifras oficiales verificadas
Fuente: Lin HM, Colak E, Richards T, et al. *The RSNA Cervical Spine Fracture CT Dataset.*
Radiology: Artificial Intelligence 2023;5(5):e230034. https://doi.org/10.1148/ryai.230034

| Dato | Valor |
|---|---|
| Total de estudios TC | 3112 |
| Positivos para fractura | 1445 (954 hombres, 491 mujeres; edad media 56.78 ± 21.97) |
| Negativos | 1667 (1022 hombres, 645 mujeres; edad media 50.61 ± 21.29) |
| Train (Kaggle) | 2019 estudios |
| Test público / privado | 304 / 789 |
| Estudios con bounding box | 235 (≈16% de los casos positivos) |
| Origen | 12 instituciones, 9 países, 6 continentes |
| Criterio de inclusión | TC axial **sin contraste, cortes de 1 mm** |
| Exclusión | pacientes con cirugía previa (artefactos metálicos y anatomía alterada) |

**Distribución de referencia:** C2, C6 y C7 son los niveles más fracturados —C7 el más
frecuente— y concentran el **62.4%** de todas las fracturas cervicales del dataset.
Esto coincide con Goldberg et al. (2001), que reportó 63.3% para esos tres niveles en un
estudio multicéntrico de 21 instituciones. **El miembro B debe contrastar su hallazgo
empírico contra este 62.4%**; si no coincide, hay un error en el cálculo.

### Semántica de las etiquetas
- `C1`..`C7`: 1 si esa vértebra cervical tiene fractura.
- `patient_overall`: 1 si hay fractura cervical de cualquier tipo.
- La verdad de referencia proviene del **informe original del radiólogo**, no de una
  relectura. Los anotadores podían disputarla y un comité adjudicaba. Los archivos DICOM
  se nombran según el atributo `InstanceNumber` (posición dentro de la pila).
- **No siempre `patient_overall == OR(C1..C7)`.** Hay estudios positivos sin nivel asignado.
  No es un error de datos: refleja incertidumbre diagnóstica real. No "corregir", documentar.

### Segmentaciones `.nii`
Valores: `1–7` = C1–C7, `8–19` = vértebras torácicas T1–T12, `0` = fondo.
No todos los estudios segmentados incluyen niveles torácicos.

### Bounding boxes
`x, y` = esquina superior izquierda. `slice_number` concatenado con `.dcm` da el nombre
del archivo. Cubren solo ~12% de los estudios.

---

## 3. Entorno y stack

```
python >= 3.10
pandas, numpy, matplotlib, seaborn, scipy
pydicom      # lectura DICOM
nibabel      # lectura NIfTI (.nii)
```

**Ejecución:** Kaggle Notebooks (el dataset se monta en
`/kaggle/input/rsna-2022-cervical-spine-fracture-detection`). Descargar 350 GB en local
no es viable. El código debe detectar el entorno:

```python
KAGGLE = os.path.exists("/kaggle/input")
BASE = ("/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
        if KAGGLE else "./dataset")
```

---

## 4. Trampas técnicas conocidas — LEER ANTES DE CODIFICAR

Estas son las que rompen el análisis de forma silenciosa (no lanzan error, solo dan
resultados equivocados):

1. **Nunca leas los DICOM completos para extraer metadatos.**
   Usa `pydicom.dcmread(path, stop_before_pixels=True)` y lee solo 3 archivos por estudio
   (primero, medio, último). Leer los 700k archivos con píxeles tarda horas.
   **Cachea el dataframe resultante en CSV** — sin caché cada reinicio del kernel cuesta minutos.

2. **El orden alfabético de los archivos NO es el orden anatómico.**
   `"10.dcm"` va antes que `"2.dcm"` en orden lexicográfico. Ordena por el entero:
   `sorted(int(f.split(".")[0]) for f in os.listdir(d))`.

3. **Hay estudios con el eje z invertido.** Unos apilan de craneal a caudal y otros al revés.
   Para trabajo anatómico serio hay que ordenar por `ImagePositionPatient[2]`, no por el
   número de archivo. Registra `z_dir` como variable categórica y reporta cuántos hay.

4. **Las máscaras NIfTI están en plano SAGITAL; los DICOM en plano AXIAL.**
   Esta es la razón oficial documentada por RSNA. Si se superponen sin reorientar, la
   máscara queda **volteada en el eje z y/o espejada en el eje x**, y parecerá que las
   anotaciones están mal.
   Lo correcto es usar la información de orientación de la cabecera NIfTI:
   ```python
   img = nib.load(f)
   img = nib.as_closest_canonical(img)   # reorienta según la matriz affine
   seg = np.asanyarray(img.dataobj)
   ```
   La transposición empírica que circula en la comunidad
   (`np.transpose(vol, (2,1,0))[::-1,::-1,:]`) funciona en este dataset, pero es frágil.
   Úsala solo como comprobación: **superpón la máscara sobre el corte y verifica
   visualmente que la vértebra coincide** antes de dar por buena cualquier alineación.

9. **No esperes gran variación en `slice_thickness`.**
   El criterio de inclusión oficial fue TC axial sin contraste de **1 mm**, así que los
   cortes deberían ser homogéneos. Si el análisis muestra mucha dispersión, es un
   hallazgo real que vale la pena reportar (variabilidad de reconstrucción entre sitios),
   no un error. La heterogeneidad esperada está más bien en `n_slices`, `pixel_spacing`
   y la cobertura anatómica `z_extent`.

5. **Los píxeles crudos no son unidades Hounsfield.**
   Aplicar siempre: `HU = pixel_array * RescaleSlope + RescaleIntercept`.

6. **Sin ventana radiológica las imágenes se ven inservibles.**
   El rango HU va de -1000 (aire) a +3000 (hueso cortical); la pantalla tiene 256 niveles.
   Ventana ósea: centro 400, ancho 1800. Ventana de tejido blando: centro 40, ancho 400.

7. **Muchos tags DICOM son opcionales.** Usa siempre `getattr(ds, tag, default)`.
   `SliceThickness`, `KVP`, `PatientSex`, `ConvolutionKernel` faltan en parte del dataset.

8. **Envuelve la extracción de metadatos en try/except por estudio** y acumula los fallos
   en una lista. Un solo estudio corrupto no debe abortar un loop de 2000 iteraciones.

---

## 5. Estructura del repositorio

```
proyecto2-eda-cervical/
├── README.md                    # descripción, cómo correr, links
├── SPECS.md                     # este documento
├── requirements.txt
├── notebooks/
│   ├── 01_etiquetas.ipynb       # Miembro B
│   ├── 02_metadatos_dicom.ipynb # Miembro B
│   ├── 03_espacial.ipynb        # Miembro C
│   └── 04_visual.ipynb          # Miembro C
├── src/
│   ├── config.py                # rutas, constantes, detección de entorno
│   ├── io_dicom.py              # lectura DICOM, HU, ventana, volúmenes
│   ├── metadata.py              # extracción y caché de la tabla de metadatos
│   ├── plots.py                 # funciones de graficado reutilizables
│   └── quality.py               # chequeos de calidad de datos
├── data/
│   └── meta_train.csv           # caché generado (SÍ commitear, pesa poco)
├── figures/                     # PNG exportados para el informe y el PPT
└── informe/
    ├── informe.md               # fuente del informe
    └── informe.pdf              # entregable
```

**Regla:** todas las figuras del informe se exportan a `figures/` con
`plt.savefig(..., dpi=200, bbox_inches="tight")`. Nada de screenshots.

---

## 6. División del trabajo

> ⚠️ **Rellenar los nombres reales antes de empezar.**
> La calificación individual sale de los commits, así que **cada quien commitea su propio
> código**. No consolidar todo desde una sola cuenta.

| Rol | Nombre | Archivos que le pertenecen |
|---|---|---|
| **A — Marco conceptual** | *(nombre)* | `informe/` secciones 1–3, `README.md` |
| **B — Datos tabulares** | *(nombre)* | `src/config.py`, `src/metadata.py`, `notebooks/01`, `notebooks/02` |
| **C — Análisis espacial y visual** | *(nombre)* | `src/io_dicom.py`, `notebooks/03`, `notebooks/04` |
| **D — Calidad, síntesis y entrega** | *(nombre)* | `src/quality.py`, `src/plots.py`, `informe/` secciones 4–6, PPT |

---

### Miembro A — Marco conceptual (30 pts de la rúbrica)

Este rol **no escribe código de análisis**, escribe el marco del informe. Es el que más
peso tiene en puntos por hora invertida.

**Tareas:**
1. **Investigación clínica.** Bibliografía semilla ya verificada (todas citadas en el
   paper oficial del dataset, así que son defendibles):

   | Dato para el informe | Fuente |
   |---|---|
   | Más de 3 millones de pacientes al año con lesión cervical en Norteamérica | Milby AH et al. *Neurosurg Focus* 2008;25(5):E10 |
   | 10–11% de las fracturas cervicales derivan en lesión de médula espinal | Fredø HL et al. *Scand J Trauma Resusc Emerg Med* 2014;22:78 |
   | Más de 1 millón de pacientes evaluados al año en EE. UU. por sospecha de lesión cervical; en adultos se usa casi exclusivamente TC, no radiografía | Minja FJ et al. *Neuroimaging Clin N Am* 2018;28(3):483–493 |
   | C2, C6 y C7 concentran 63.3% de las fracturas (21 instituciones) | Goldberg W et al. *Ann Emerg Med* 2001;38(1):17–21 |
   | Un algoritmo entrenado en datos multicéntricos tuvo precisión diagnóstica muy limitada al aplicarse a datos externos | Voter AF et al. *AJNR* 2021;42(8):1550–1556 |
   | Descripción oficial del dataset | Lin HM et al. *Radiol Artif Intell* 2023;5(5):e230034 |

   Puntos a desarrollar con esas fuentes:
   - Qué es una fractura cervical, incidencia, mecanismos de lesión.
   - Consecuencias del diagnóstico tardío (deterioro neurológico, parálisis).
   - **Por qué la TC desplazó a la radiografía simple** en adultos — esto además
     justifica la aclaración sobre el error del enunciado.
   - Por qué la interpretación es difícil: en población mayor, la enfermedad
     degenerativa y la osteoporosis superpuestas confunden la detección.
   - **Argumento de generalización:** los modelos previos se entrenaron con
     representación geográfica limitada y fallaron en datos externos. Ese es el vacío
     que este dataset multinacional busca llenar. Es un excelente cierre para la
     situación problemática.
2. **Situación problemática** (10 pts): redactar el contexto que da origen al problema.
3. **Problema científico** (10 pts): enunciarlo con precisión, en una pregunta.
4. **Objetivos** (10 pts): 1 general + mínimo 3 específicos.
   Deben ser **medibles y alcanzables dentro de un análisis exploratorio** — ojo con
   redactar objetivos de modelado, porque este proyecto no entrena modelos.
5. `README.md` del repo.

**Criterio de aceptación:** cada afirmación clínica tiene cita. Los objetivos son
verificables contra el contenido real del informe.

---

### Miembro B — Datos tabulares y metadatos (parte del rubro de 20 + 30 pts)

**Tareas:**

`src/config.py`
- Detección de entorno, rutas, constantes (`LEVELS = ["C1",...,"C7"]`, paletas, dpi).

`src/metadata.py`
- `meta_estudio(uid) -> dict` — extrae de 3 cabeceras DICOM por estudio:
  `n_slices, slice_gap, slice_thickness, pixel_spacing, rows, cols, z_first, z_last,
  z_extent, z_dir, kvp, manufacturer, model, kernel, patient_sex, patient_age, study_desc`.
- `construir_tabla(uids, cache="data/meta_train.csv") -> DataFrame` — con caché,
  barra de progreso y captura de errores por estudio.

`notebooks/01_etiquetas.ipynb`
- Conteo de observaciones y variables; tipos; nulos; duplicados; integridad
  (UIDs de `train.csv` sin carpeta de imágenes).
- Chequeo `patient_overall vs OR(C1..C7)` y cuantificación de las inconsistencias.
- Tabla de frecuencias y proporciones de las 8 etiquetas + gráfico de barras.
- Distribución del número de niveles fracturados por paciente.
- Matriz de correlación phi entre C1–C7 y matriz de co-ocurrencia (heatmaps).
- Top 15 combinaciones de niveles fracturados.
- **Validación contra la referencia oficial:** calcular qué porcentaje del total de
  fracturas aportan C2 + C6 + C7. Debe dar cerca de **62.4%**, y C7 debe salir como el
  nivel más frecuente. Si no coincide, revisar el cálculo antes de seguir.

`notebooks/02_metadatos_dicom.ipynb`
- `describe()` de las variables numéricas.
- Histogramas y boxplots de `n_slices, slice_thickness, pixel_spacing, z_extent`.
- Outliers por IQR: cuantificar, **inspeccionar los casos extremos y argumentar por qué
  no se eliminan** (son variabilidad legítima de protocolo entre 12 hospitales).
- Matriz de correlación de Pearson entre variables numéricas + scatter
  `slice_thickness` vs `n_slices` (esperar relación hiperbólica, no lineal —
  comentar que `z_extent ≈ n_slices × slice_thickness`).
- Tablas de frecuencia de las categóricas (`manufacturer`, `z_dir`, `kernel`, `patient_sex`).
- **Cruce con la etiqueta:** chi² fabricante vs `patient_overall`; Mann-Whitney de
  `n_slices`/`slice_thickness`/`z_extent` entre positivos y negativos; boxplots por clase.
  Interpretación: si hay asociación, existe riesgo de *shortcut learning* (el modelo
  aprende a reconocer el escáner en vez de la fractura).

**Criterio de aceptación:** `data/meta_train.csv` commiteado, y cada gráfico tiene debajo
un párrafo interpretando qué se ve. Gráfico sin interpretación no cuenta.

---

### Miembro C — Análisis espacial y visual (parte del rubro de 30 pts)

**Tareas:**

`src/io_dicom.py`
- `cargar_hu(path)` — lee DICOM y aplica `RescaleSlope`/`RescaleIntercept`.
- `ventana(hu, centro, ancho)` — normaliza a [0,1].
- `cargar_volumen(uid, paso=1)` — apila cortes **ordenados por posición z real**.
- `cargar_segmentacion(uid)` — carga `.nii` y aplica la transposición/volteo del punto 4
  de la sección de trampas.

`notebooks/03_espacial.ipynb`
- Bounding boxes: cobertura (% de estudios anotados), cajas por estudio, distribución del
  área de la lesión, scatter ancho vs alto.
- **Área de la lesión como % de la imagen 512×512.** Este es el hallazgo central del
  reto: la fractura ocupa una fracción minúscula del corte.
- Posición relativa de la fractura dentro del volumen (`slice_number / n_slices`).
- Solapamiento entre estudios con caja y estudios con segmentación.
- Segmentaciones: volumen en voxeles por vértebra C1–C7, % de estudios que incluyen
  niveles torácicos, gráfico de barras del volumen medio por vértebra.

`notebooks/04_visual.ipynb`
- Demostración del efecto de la ventana: mismo corte sin ventana / tejido blando / ósea.
- Superposición de bounding box sobre el corte correspondiente.
- Reconstrucción sagital y coronal (la vista sagital es donde el radiólogo evalúa la
  alineación cervical).
- Superposición de máscara de segmentación con transparencia, coloreada por vértebra.
- Rejilla comparativa de cortes axiales de casos positivos vs negativos, para mostrar
  que la diferencia **no es evidente a simple vista**.

**Criterio de aceptación:** todas las figuras exportadas a `figures/` en 200 dpi.
La superposición de la máscara debe estar visiblemente alineada con la anatomía —
si se ve al revés, la transposición está mal.

---

### Miembro D — Calidad, síntesis y entrega (20 pts + descripción de datos)

**Tareas:**

`src/quality.py`
- `reporte_calidad(train, meta, bbox, seg_uids) -> DataFrame` con una fila por problema
  detectado y su magnitud cuantificada:
  - inconsistencia `patient_overall` vs niveles
  - huecos en la numeración de cortes (`slice_gap > 0`)
  - estudios con eje z descendente
  - metadatos ausentes por columna
  - número de valores distintos de `pixel_spacing` y `slice_thickness`
  - combinaciones distintas de `rows × cols`
  - cobertura de anotación espacial (cajas y segmentaciones)

`src/plots.py`
- Funciones de graficado reutilizables + `guardar(fig, nombre)` que exporta a `figures/`
  con dpi y estilo consistentes. **B y C deben usarlas** para que las figuras del informe
  se vean homogéneas.

**Informe, sección "Descripción de los datos" (20 pts):**
- Tabla de variables con tipo y descripción (las 8 etiquetas + las ~17 de metadatos).
- Número de observaciones y variables.
- Descripción de las operaciones de limpieza y preprocesamiento **derivadas de los
  hallazgos**, no genéricas:
  1. Normalizar orientación (ordenar por posición z; transponer NIfTI).
  2. Convertir a HU y remuestrear a espaciado isotrópico 1×1×1 mm.
  3. Aplicar ventana ósea y descartar el rango de aire.
  4. Recortar a la región C1–C7 usando las segmentaciones.
  5. Decidir el tratamiento de las etiquetas ambiguas y justificarlo.
  6. Partición estratificada por `patient_overall` y por fabricante.

**Informe, sección "Hallazgos y conclusiones" (20 pts):**
- Resumen de hallazgos con **los números reales**, no plantilla.
- Conclusiones sobre los siguientes pasos, aterrizadas en lo que mostró el EDA
  (por ejemplo: el tamaño diminuto de la lesión justifica un enfoque en dos etapas —
  localizar vértebras primero, clasificar la vértebra recortada después).

**Entrega:**
- Consolidar `informe/informe.md` → PDF.
- Presentación PowerPoint (~12 slides: problema, datos, 5–6 hallazgos con figura,
  conclusiones).
- Verificar que el repo sea público o que el catedrático tenga acceso.

---

## 7. Convenciones de Git

```
main                    # solo merges revisados
feat/<inicial>-<tema>   # ej: feat/b-metadatos, feat/c-segmentacion
```

- Commits en español, imperativo: `agrega extracción de metadatos DICOM con caché`.
- **Cada miembro commitea su propio trabajo desde su propia cuenta.** La nota individual
  depende de esto.
- No commitear DICOM ni `.nii`. `.gitignore` debe incluir `*.dcm`, `*.nii`, `*.nii.gz`,
  `.ipynb_checkpoints/`. Sí commitear `data/meta_train.csv` (pesa poco y ahorra horas).
- Limpiar outputs pesados de los notebooks antes de commitear, salvo las figuras finales.

---

## 8. Cronograma (entrega 11 de septiembre)

| Día | A | B | C | D |
|---|---|---|---|---|
| 7 sep | Investigación clínica | `config.py`, `metadata.py`, correr extracción | `io_dicom.py` | Estructura del repo, `plots.py` |
| 8 sep | Situación + problema | Notebook 01 | Notebook 03 | `quality.py` |
| 9 sep | Objetivos | Notebook 02 | Notebook 04 | Descripción de datos |
| 10 sep | Revisión cruzada | Interpretaciones | Exportar figuras | Hallazgos + conclusiones |
| 11 sep | — | — | — | PDF + PPT + entrega |

**Ruta crítica:** la extracción de metadatos del miembro B bloquea a D. Debe correr el
día 1 y quedar cacheada en el repo.

---

## 9. Mapa a la rúbrica

| Puntos | Rubro | Responsable |
|---|---|---|
| 10 | Situación problemática | A |
| 10 | Problema científico | A |
| 10 | Objetivos | A |
| 20 | Descripción de los datos | D (con insumos de B) |
| 30 | Análisis exploratorio | B + C |
| 20 | Hallazgos y conclusiones | D |

**Lo que más se penaliza:** gráficos sin interpretación escrita. La rúbrica dice
literalmente "explica muy bien todos los procedimientos y los hallazgos". Cada figura
necesita un párrafo debajo que diga qué se ve y qué implica.
