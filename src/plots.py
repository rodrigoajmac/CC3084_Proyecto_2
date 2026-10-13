"""
src/plots.py
Funciones de graficado reutilizables para el Proyecto 2 (Miembro D).

Objetivo: que las figuras producidas por los miembros B y C se vean homogéneas
en el informe y en la presentación. Todas las figuras se exportan con
`guardar()`, que centraliza dpi, recorte y fondo.

Uso típico desde un notebook:

    from src.plots import aplicar_estilo, guardar, interpretar
    aplicar_estilo()
    fig, ax = plt.subplots()
    ...
    guardar(fig, "03_area_lesion.png")
    interpretar("Qué se ve en la figura y qué implica para el proyecto.")

El módulo no depende de `src.config` de forma estricta: si no puede importarlo
(por ejemplo, en un notebook de Kaggle donde solo se subió este archivo),
cae a valores por defecto equivalentes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuración: se intenta reutilizar src/config.py; si no existe, hay respaldo
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depende del entorno de ejecución
    from src.config import DPI, FIGURES_DIR, LEVELS, PALETTE_DIVERGING, PALETTE_NAME
except Exception:  # pragma: no cover
    DPI = 200
    PALETTE_NAME = "crest"
    PALETTE_DIVERGING = "vlag"
    LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    _kaggle = Path("/kaggle/input").exists()
    FIGURES_DIR = Path("/kaggle/working/figures") if _kaggle else Path("figures")

FIGURES_DIR = Path(FIGURES_DIR)

# Paleta cualitativa fija del proyecto. Se usa siempre en el mismo orden para que
# un mismo concepto conserve su color entre figuras distintas.
COLORES = {
    "primario": "#264653",
    "secundario": "#2a9d8f",
    "acento": "#e76f51",
    "neutro": "#457b9d",
    "resalte": "#f4a261",
    "alterno": "#6a4c93",
}
COLOR_POSITIVO = COLORES["acento"]
COLOR_NEGATIVO = COLORES["neutro"]

# Colores por vértebra: los mismos en barras, heatmaps y máscaras de segmentación.
COLORES_VERTEBRAS = {
    "C1": "#e63946",
    "C2": "#f4a261",
    "C3": "#e9c46a",
    "C4": "#2a9d8f",
    "C5": "#00b4d8",
    "C6": "#4361ee",
    "C7": "#7209b7",
}


def aplicar_estilo() -> None:
    """Fija el estilo visual común a todos los notebooks del proyecto."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,          # cómodo en pantalla
        "savefig.dpi": DPI,         # calidad de informe
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "font.size": 10,
    })


def guardar(fig: plt.Figure, nombre: str, carpeta: Path | str | None = None) -> Path:
    """Exporta una figura a `figures/` con los parámetros del informe.

    Es la única vía autorizada para producir figuras del entregable: garantiza
    200 dpi, recorte ajustado y fondo blanco (evita que el fondo transparente se
    vea negro al pegar la imagen en el PowerPoint).
    """
    destino = Path(carpeta) if carpeta is not None else FIGURES_DIR
    destino.mkdir(parents=True, exist_ok=True)
    if not nombre.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
        nombre = f"{nombre}.png"
    ruta = destino / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Figura guardada: {ruta}")
    return ruta


def interpretar(texto: str) -> None:
    """Imprime la interpretación obligatoria que acompaña a cada figura.

    La rúbrica penaliza el gráfico sin explicación escrita, así que esta función
    existe para que el paso sea explícito y difícil de olvidar.
    """
    try:  # dentro de Jupyter se renderiza como markdown
        from IPython.display import Markdown, display

        display(Markdown(f"**Interpretación.** {texto}"))
    except Exception:  # pragma: no cover - fuera de Jupyter
        print(f"Interpretación: {texto}")


# ---------------------------------------------------------------------------
# Gráficos reutilizables
# ---------------------------------------------------------------------------
def barras_frecuencia(
    serie: pd.Series,
    titulo: str,
    xlabel: str = "",
    ylabel: str = "Estudios",
    color: str | None = None,
    horizontal: bool = False,
    anotar: bool = True,
    figsize: tuple[float, float] = (9, 4.8),
) -> plt.Figure:
    """Barras de conteo con etiquetas de valor sobre cada barra.

    `serie` debe venir ya agregada (índice = categoría, valor = conteo).
    """
    fig, ax = plt.subplots(figsize=figsize)
    color = color or COLORES["primario"]
    if horizontal:
        sns.barplot(x=serie.values, y=serie.index.astype(str), color=color, ax=ax)
        ax.set(title=titulo, xlabel=ylabel, ylabel=xlabel)
    else:
        sns.barplot(x=serie.index.astype(str), y=serie.values, color=color, ax=ax)
        ax.set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    if anotar and ax.containers:
        ax.bar_label(ax.containers[0], fmt="%.0f", padding=2, fontsize=9)
    fig.tight_layout()
    return fig


def barras_por_vertebra(
    serie: pd.Series,
    titulo: str,
    ylabel: str = "Estudios",
    anotar: bool = True,
    figsize: tuple[float, float] = (9, 4.8),
) -> plt.Figure:
    """Barras C1–C7 con el color fijo de cada vértebra en todo el proyecto."""
    serie = serie.reindex([nivel for nivel in LEVELS if nivel in serie.index])
    colores = [COLORES_VERTEBRAS.get(str(nivel), COLORES["primario"]) for nivel in serie.index]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(serie.index.astype(str), serie.values, color=colores)
    ax.set(title=titulo, xlabel="Nivel cervical", ylabel=ylabel)
    if anotar:
        ax.bar_label(ax.containers[0], fmt="%.0f", padding=2, fontsize=9)
    fig.tight_layout()
    return fig


def histograma_y_boxplot(
    df: pd.DataFrame,
    columna: str,
    titulo: str | None = None,
    xlabel: str | None = None,
    bins: int = 40,
    figsize: tuple[float, float] = (12, 4.2),
) -> plt.Figure:
    """Histograma + boxplot de una variable numérica, lado a lado.

    Es la vista mínima para justificar decisiones sobre outliers: el histograma
    muestra la forma y el boxplot cuantifica los extremos por IQR.
    """
    datos = pd.to_numeric(df[columna], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(datos, bins=bins, kde=True, color=COLORES["neutro"], ax=axes[0])
    axes[0].set(title=f"Distribución de {columna}", xlabel=xlabel or columna, ylabel="Estudios")
    sns.boxplot(x=datos, color=COLORES["resalte"], ax=axes[1])
    axes[1].set(title=f"Dispersión de {columna}", xlabel=xlabel or columna)
    if titulo:
        fig.suptitle(titulo, y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def heatmap_matriz(
    matriz: pd.DataFrame,
    titulo: str,
    fmt: str = ".2f",
    diverging: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (7.5, 6),
) -> plt.Figure:
    """Heatmap anotado para matrices de correlación o de co-ocurrencia."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matriz,
        annot=True,
        fmt=fmt,
        cmap=PALETTE_DIVERGING if diverging else PALETTE_NAME,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def boxplot_por_clase(
    df: pd.DataFrame,
    columnas: Sequence[str],
    clase: str = "patient_overall",
    etiquetas: tuple[str, str] = ("Sin fractura", "Con fractura"),
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Boxplots de varias variables numéricas separadas por la etiqueta.

    Sirve para inspeccionar visualmente el riesgo de *shortcut learning*: si una
    variable de adquisición separa las clases, el modelo podría aprender el
    protocolo de escaneo en lugar de la lesión.
    """
    columnas = [c for c in columnas if c in df.columns]
    if not columnas:
        raise ValueError("Ninguna de las columnas solicitadas existe en el DataFrame")
    figsize = figsize or (5 * len(columnas), 4.6)
    fig, axes = plt.subplots(1, len(columnas), figsize=figsize, squeeze=False)
    paleta = {0: COLOR_NEGATIVO, 1: COLOR_POSITIVO}
    for ax, col in zip(axes[0], columnas):
        sns.boxplot(data=df, x=clase, y=col, hue=clase, palette=paleta,
                    legend=False, ax=ax)
        ax.set(title=f"{col} por clase", xlabel="", ylabel=col)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(etiquetas)
    fig.tight_layout()
    return fig


def barras_apiladas_calidad(
    reporte: pd.DataFrame,
    titulo: str = "Problemas de calidad detectados",
    columna_valor: str = "porcentaje",
    columna_etiqueta: str = "hallazgo",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Barras horizontales del reporte de calidad, coloreadas por severidad.

    Espera el DataFrame que devuelve `src.quality.reporte_calidad`.
    """
    datos = reporte.dropna(subset=[columna_valor]).copy()
    datos = datos.sort_values(columna_valor, ascending=True)
    mapa_color = {
        "alta": COLORES["acento"],
        "media": COLORES["resalte"],
        "baja": COLORES["secundario"],
        "informativa": COLORES["neutro"],
    }
    colores = [mapa_color.get(str(s).lower(), COLORES["neutro"])
               for s in datos.get("severidad", pd.Series(index=datos.index, dtype=object))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(datos[columna_etiqueta].astype(str), datos[columna_valor], color=colores)
    ax.set(title=titulo, xlabel="Porcentaje de estudios afectados (%)", ylabel="")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlim(0, max(100, float(datos[columna_valor].max()) * 1.18))

    import matplotlib.patches as mpatches

    presentes = [s for s in ["alta", "media", "baja", "informativa"]
                 if s in set(datos.get("severidad", pd.Series(dtype=object)).astype(str).str.lower())]
    if presentes:
        ax.legend(handles=[mpatches.Patch(color=mapa_color[s], label=s.capitalize())
                           for s in presentes],
                  title="Severidad", loc="lower right")
    fig.tight_layout()
    return fig


def barras_cobertura(
    etiquetas: Iterable[str],
    valores: Iterable[float],
    titulo: str,
    ylabel: str = "Porcentaje de estudios (%)",
    figsize: tuple[float, float] = (8, 4.5),
) -> plt.Figure:
    """Barras de cobertura (porcentajes) con referencia al 100%."""
    etiquetas = list(etiquetas)
    valores = list(valores)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(etiquetas, valores, color=COLORES["secundario"])
    ax.axhline(100, color=COLORES["acento"], linestyle="--", linewidth=1,
               label="Cobertura total (100%)")
    ax.set(title=titulo, ylabel=ylabel, ylim=(0, 112))
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3, fontsize=9)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def tabla_a_figura(
    df: pd.DataFrame,
    titulo: str,
    figsize: tuple[float, float] | None = None,
    max_filas: int = 20,
) -> plt.Figure:
    """Renderiza un DataFrame pequeño como figura, para pegarlo en el PPT."""
    datos = df.head(max_filas)
    figsize = figsize or (min(2.1 * len(datos.columns) + 1.5, 16), 0.45 * len(datos) + 1.4)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tabla = ax.table(
        cellText=np.asarray(datos.astype(str).values),
        colLabels=list(datos.columns),
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.45)
    for (fila, _), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_facecolor(COLORES["primario"])
            celda.set_text_props(color="white", fontweight="bold")
        elif fila % 2 == 0:
            celda.set_facecolor("#f2f4f5")
    ax.set_title(titulo, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig
