"""
src/extraer_figuras.py
Recupera las figuras incrustadas en los notebooks y las escribe en figures/.

Uso:
    python src/extraer_figuras.py            # extrae todas las que falten
    python src/extraer_figuras.py --forzar   # reescribe las que ya existen
    python src/extraer_figuras.py --listar   # solo muestra qué hay, sin escribir

Por qué existe (Miembro D): los notebooks se ejecutan en Kaggle, donde
`guardar()` escribe en `/kaggle/working/figures`. Ese directorio no viaja con el
`.ipynb` al hacer commit, pero **los PNG sí quedan incrustados en las salidas del
notebook**. Este script los recupera de ahí, de modo que `figures/` se puede
reconstruir desde el repositorio sin volver a ejecutar nada en Kaggle ni
descargar los 350 GB del conjunto.

Las imágenes salen tal cual se generaron, a 200 dpi: no se reescalan ni se
recomprimen.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = RAIZ / "notebooks"
FIGURAS = RAIZ / "figures"

# Mapa explícito notebook -> {índice de celda: nombre de archivo}.
# Se fija a mano en lugar de numerar automáticamente para que el nombre del
# archivo no cambie si alguien inserta una celda en medio del notebook.
MAPA: dict[str, dict[int, str]] = {
    "01-etiquetas.ipynb": {
        11: "01_frecuencia_etiquetas.png",
        13: "01_correlacion_y_coocurrencia.png",
    },
    "02-metadatos-dicom.ipynb": {
        11: "02_distribuciones_metadatos.png",
        14: "02_correlacion_y_dispersion.png",
        18: "02_metadatos_por_clase.png",
    },
    "03-espacial.ipynb": {
        6: "03_area_y_forma_bounding_boxes.png",
        8: "03_cajas_por_estudio.png",
        10: "03_posicion_relativa_lesion.png",
        12: "03_solapamiento_anotaciones.png",
        14: "03_volumen_medio_vertebras.png",
    },
    "04-visual.ipynb": {
        5: "04_comparacion_ventanas.png",
        7: "04_bounding_box_corte.png",
        9: "04_reconstruccion_multiplanar.png",
        11: "04_superposicion_segmentacion.png",
        13: "04_positivos_vs_negativos.png",
    },
    "05-calidad.ipynb": {},  # se rellena cuando el notebook se ejecute
}


def _png_de_celda(celda: dict) -> bytes | None:
    """Devuelve el primer PNG incrustado en las salidas de una celda."""
    for salida in celda.get("outputs", []):
        datos = salida.get("data", {})
        if "image/png" in datos:
            crudo = datos["image/png"]
            return base64.b64decode("".join(crudo) if isinstance(crudo, list) else crudo)
    return None


def extraer(forzar: bool = False, listar: bool = False) -> int:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    escritas = omitidas = faltantes = 0

    for nombre_nb, celdas in MAPA.items():
        ruta_nb = NOTEBOOKS / nombre_nb
        if not ruta_nb.exists():
            print(f"[!] No existe {ruta_nb.relative_to(RAIZ)}")
            continue
        if not celdas:
            continue

        notebook = json.loads(ruta_nb.read_text(encoding="utf-8"))
        total = len(notebook["cells"])

        for indice, nombre_png in celdas.items():
            destino = FIGURAS / nombre_png
            if indice >= total:
                print(f"[!] {nombre_nb}: la celda {indice} ya no existe "
                      f"(el notebook tiene {total}). Revisa el MAPA de este script.")
                faltantes += 1
                continue

            imagen = _png_de_celda(notebook["cells"][indice])
            if imagen is None:
                print(f"[!] {nombre_nb} celda {indice}: sin PNG en las salidas. "
                      f"Falta {nombre_png} — ejecuta el notebook en Kaggle y guárdalo con sus salidas.")
                faltantes += 1
                continue

            if listar:
                estado = "ya existe" if destino.exists() else "pendiente"
                print(f"    {nombre_png:45s} {len(imagen)/1024:7.1f} KB  ({estado})")
                continue

            if destino.exists() and not forzar:
                omitidas += 1
                continue

            destino.write_bytes(imagen)
            print(f"    {nombre_png:45s} {len(imagen)/1024:7.1f} KB")
            escritas += 1

    if not listar:
        print(f"\nEscritas: {escritas} · Ya existían: {omitidas} · Sin PNG en el notebook: {faltantes}")
        if omitidas and not forzar:
            print("Usa --forzar para reescribir las que ya existen.")
    return 1 if faltantes else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forzar", action="store_true",
                        help="reescribe las figuras que ya están en figures/")
    parser.add_argument("--listar", action="store_true",
                        help="muestra qué figuras hay incrustadas, sin escribir nada")
    argumentos = parser.parse_args()
    sys.exit(extraer(forzar=argumentos.forzar, listar=argumentos.listar))
