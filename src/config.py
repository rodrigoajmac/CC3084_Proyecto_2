"""
src/config.py
Configuración global, constantes y detección de rutas para el proyecto.
"""

from pathlib import Path
import os

# -------------------------------------------------------------------------
# Detección de entorno y rutas base
# -------------------------------------------------------------------------
KAGGLE = os.path.exists("/kaggle/input")

if KAGGLE:
    BASE_DIR = Path("/kaggle/input/rsna-2022-cervical-spine-fracture-detection")
    PROJECT_ROOT = Path("/kaggle/working")
else:
    # Asume ejecución desde la raíz del repositorio o subcarpeta
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    BASE_DIR = PROJECT_ROOT / "dataset"

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Asegurar existencia de directorios de salida
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Rutas a archivos de entrada
TRAIN_CSV = BASE_DIR / "train.csv"
TRAIN_BBOX_CSV = BASE_DIR / "train_bounding_boxes.csv"
TEST_CSV = BASE_DIR / "test.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images"
SEGMENTATIONS_DIR = BASE_DIR / "segmentations"
META_CACHE_FILE = DATA_DIR / "meta_train.csv"

# -------------------------------------------------------------------------
# Constantes clínicas y de modelado
# -------------------------------------------------------------------------
LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
ALL_TARGETS = LEVELS + ["patient_overall"]

# -------------------------------------------------------------------------
# Estilo de gráficos (acorde a la regla de figuras para informe)
# -------------------------------------------------------------------------
DPI = 200
PALETTE_NAME = "crest"
PALETTE_DIVERGING = "vlag"