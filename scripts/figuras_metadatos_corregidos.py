"""
scripts/figuras_metadatos_corregidos.py
Regenera las figuras de metadatos con el z_extent FÍSICO (Miembro D).

Por qué existe: las figuras del notebook 02 calculan `z_extent` como
`n_slices × slice_thickness`, que sobreestima el recorrido anatómico en el
99.9 % de los estudios. El texto del informe usa `|z_last − z_first|` tomado de
`ImagePositionPatient[2]`, así que las figuras y el texto no coincidían.

Este script cierra esa brecha usando `data/meta_train.csv` (esquema completo,
generado en Kaggle por notebooks/05-calidad.ipynb) y las funciones de
`src/plots.py`, para que el estilo sea el mismo que el del resto del informe.

Uso:
    python scripts/figuras_metadatos_corregidos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.plots import COLOR_NEGATIVO, COLOR_POSITIVO, COLORES, aplicar_estilo, guardar

META = RAIZ / "data" / "meta_train.csv"
TRAIN = RAIZ / "data" / "raw" / "train.csv"


def main() -> None:
    if not META.exists():
        raise SystemExit(
            f"Falta {META}. Ejecuta notebooks/05-calidad.ipynb en Kaggle y descarga "
            "data/meta_train.csv antes de regenerar estas figuras."
        )
    aplicar_estilo()
    meta = pd.read_csv(META)
    aprox = meta["n_slices"] * meta["slice_thickness"]
    fisico = meta["z_extent"]

    # ---------------------------------------------------------------------
    # Figura 1: la aproximación frente al recorrido físico
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    bins = np.linspace(0, 700, 60)
    axes[0].hist(aprox, bins=bins, alpha=.72, label="n_slices × grosor",
                 color=COLORES["acento"])
    axes[0].hist(fisico, bins=bins, alpha=.72, label="|z_last − z_first| (real)",
                 color=COLORES["secundario"])
    axes[0].set(title="Cobertura craneocaudal: aproximación vs. medida real",
                xlabel="z_extent (mm)", ylabel="Estudios")
    axes[0].legend()

    axes[1].scatter(fisico, aprox, s=9, alpha=.28, color=COLORES["alterno"])
    limite = float(max(aprox.max(), fisico.max()))
    axes[1].plot([0, limite], [0, limite], "--", color="0.35", lw=1.2,
                 label="identidad (sin solapamiento)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set(title="Casi siempre por encima de la identidad",
                xlabel="Recorrido físico (mm)", ylabel="n_slices × grosor (mm)")
    axes[1].legend()

    exceso = aprox - fisico
    axes[2].hist(exceso, bins=60, color=COLORES["resalte"])
    axes[2].axvline(float(exceso.median()), color=COLORES["primario"], ls="--", lw=1.4,
                    label=f"Mediana = {exceso.median():.1f} mm")
    axes[2].set(title="Sobreestimación = solapamiento entre cortes",
                xlabel="Exceso de la aproximación (mm)", ylabel="Estudios")
    axes[2].legend()

    fig.suptitle("Corrección de z_extent: el recorrido real es mucho más homogéneo",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    guardar(fig, "05_z_extent_corregido.png")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figura 2: distribuciones con las variables correctas
    # ---------------------------------------------------------------------
    variables = [
        ("n_slices", "Cortes por estudio"),
        ("slice_thickness", "Grosor de corte (mm)"),
        ("pixel_spacing_x", "Espaciado de píxel (mm)"),
        ("z_extent", "Recorrido craneocaudal real (mm)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2),
                             gridspec_kw={"height_ratios": [3, 1]})
    for indice, (columna, etiqueta) in enumerate(variables):
        datos = pd.to_numeric(meta[columna], errors="coerce").dropna()
        axes[0, indice].hist(datos, bins=45, color=COLORES["neutro"])
        axes[0, indice].set(title=etiqueta, ylabel="Estudios" if indice == 0 else "")
        axes[1, indice].boxplot(datos, orientation="horizontal", widths=.55,
                                patch_artist=True,
                                boxprops=dict(facecolor=COLORES["resalte"],
                                              color=COLORES["primario"]),
                                medianprops=dict(color=COLORES["primario"], lw=1.6))
        axes[1, indice].set(yticks=[], xlabel=etiqueta)
    fig.suptitle("Metadatos de adquisición con el z_extent medido por cabecera",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    guardar(fig, "05_distribuciones_corregidas.png")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figura 3: cruce con la etiqueta usando las variables correctas
    # ---------------------------------------------------------------------
    if TRAIN.exists():
        from scipy.stats import mannwhitneyu

        train = pd.read_csv(TRAIN)
        d = meta.merge(train[["StudyInstanceUID", "patient_overall"]],
                       on="StudyInstanceUID")
        columnas = ["n_slices", "slice_thickness", "z_extent"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, columna in zip(axes, columnas):
            pos = d.loc[d.patient_overall.eq(1), columna].dropna()
            neg = d.loc[d.patient_overall.eq(0), columna].dropna()
            estadistico, p = mannwhitneyu(pos, neg, alternative="two-sided")
            cles = estadistico / (len(pos) * len(neg))
            caja = ax.boxplot([neg, pos], patch_artist=True, widths=.55,
                              medianprops=dict(color="white", lw=1.8))
            for parche, color in zip(caja["boxes"], [COLOR_NEGATIVO, COLOR_POSITIVO]):
                parche.set_facecolor(color)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Sin fractura", "Con fractura"])
            ax.set(title=f"{columna}\np = {p:.2e} · P(pos>neg) = {cles:.3f}",
                   ylabel=columna)
        fig.suptitle("Riesgo de shortcut learning con las variables corregidas",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        guardar(fig, "05_metadatos_por_clase_corregido.png")
        plt.close(fig)
    else:
        print(f"AVISO: falta {TRAIN}; se omite la figura de cruce con la etiqueta.")

    print("\nResumen de la corrección:")
    print(f"  aproximación : mediana {aprox.median():7.1f} mm | sigma {aprox.std():6.1f} | "
          f"rango {aprox.min():.1f}-{aprox.max():.1f}")
    print(f"  real         : mediana {fisico.median():7.1f} mm | sigma {fisico.std():6.1f} | "
          f"rango {fisico.min():.1f}-{fisico.max():.1f}")
    print(f"  sobreestima en {100 * (exceso > 0).mean():.1f}% de los estudios")


if __name__ == "__main__":
    main()
