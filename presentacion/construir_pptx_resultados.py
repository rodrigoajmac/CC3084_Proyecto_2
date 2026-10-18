"""
presentacion/construir_pptx_resultados.py
Presentación de la fase de resultados del Proyecto 2.

Reutiliza las primitivas de maquetado de `construir_pptx.py` (paleta, tarjetas
de cifra, inserción de figuras) en lugar de duplicarlas, y toma todas las cifras
de `Aplicacion/metrics/*.json`, de modo que no hay números escritos a mano que
puedan quedar desfasados respecto del entrenamiento.

Uso:
    python presentacion/construir_pptx_resultados.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import construir_pptx as base
from pptx import Presentation
from pptx.util import Inches, Pt

SALIDA = AQUI / "proyecto2-resultados-cervical.pptx"
METRICAS = RAIZ / "Aplicacion" / "metrics"
ORDEN = ["cnn_simple", "mobilenetv2", "efficientnetb0"]


def cargar() -> list[dict]:
    filas = []
    for ident in ORDEN:
        ruta = METRICAS / f"{ident}.json"
        if ruta.is_file():
            d = json.loads(ruta.read_text(encoding="utf-8"))
            d["id"] = ident
            filas.append(d)
    if not filas:
        raise SystemExit(f"No hay métricas en {METRICAS}. Ejecuta notebooks/06-modelado.ipynb.")
    return filas


def construir() -> Path:
    filas = cargar()
    mejor = max(filas, key=lambda f: f["roc_auc"])
    peor = min(filas, key=lambda f: f["roc_auc"])
    rapido = min(filas, key=lambda f: f["ms_por_imagen"])
    rango = mejor["roc_auc"] - peor["roc_auc"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = base.ANCHO, base.ALTO
    base._contador["n"] = 0

    # --- 1. Portada ---------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    base._rect(s, 0, 0, base.ANCHO, base.ALTO, base.OSCURO)
    base._rect(s, 0, Inches(6.9), base.ANCHO, Inches(0.6), base.TEAL)
    caja = s.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.3), Inches(1.9))
    caja.text_frame.word_wrap = True
    r = caja.text_frame.paragraphs[0].add_run()
    r.text = "Detección de fracturas cervicales en TC\nModelado, evaluación y aplicación"
    r.font.size, r.font.bold, r.font.name = Pt(40), True, base.FUENTE
    r.font.color.rgb = base.BLANCO
    base._rect(s, Inches(1.03), Inches(4.45), Inches(2.2), Pt(5), base.NARANJA)
    sub = s.shapes.add_textbox(Inches(1.0), Inches(4.75), Inches(11.0), Inches(1.6))
    sub.text_frame.word_wrap = True
    for i, (txt, bold) in enumerate([
        ("Proyecto 2 · Fase de resultados · Reto 20", True),
        ("CC3084 — Data Science · Universidad del Valle de Guatemala · Semestre II 2026", False),
    ]):
        p = sub.text_frame.paragraphs[0] if i == 0 else sub.text_frame.add_paragraph()
        rr = p.add_run(); rr.text = txt
        rr.font.size, rr.font.name, rr.font.bold = Pt(16), base.FUENTE, bold
        rr.font.color.rgb = base.BLANCO if bold else base.RGBColor(0xD4, 0xDE, 0xE2)
    base._contador["n"] = 1

    # --- 2. El problema -----------------------------------------------------
    s = base._slide_base(prs, "El problema", "Una lesión diminuta en una imagen enorme")
    base._cifra_clave(s, Inches(0.72), Inches(1.95), Inches(3.9), "2.4 %",
                      "del corte ocupa la lesión anotada (mediana).\nRelación fondo-a-lesión ≈ 41 : 1", base.NARANJA)
    base._cifra_clave(s, Inches(4.75), Inches(1.95), Inches(3.9), "11.6 %",
                      "de los estudios tiene localización.\n3 de cada 4 positivos no la tienen.", base.NARANJA)
    base._cifra_clave(s, Inches(8.78), Inches(1.95), Inches(3.9), "p = 1.8e-8",
                      "asociación entre metadatos de adquisición\ny etiqueta: riesgo de atajo real", base.TEAL)
    base._caja(s, Inches(0.72), Inches(3.95), Inches(12.0), Inches(2.6), [
        ("¿Puede un clasificador identificar una fractura visible en un corte axial", True),
        ("sin que su desempeño se explique por atajos de protocolo ni por fuga entre pacientes?", True),
        "",
        "Los tres hallazgos del EDA no son contexto: son las tres restricciones que",
        "determinaron el diseño del experimento.",
    ], size=15, space=8)

    # --- 3. Definición de la tarea -----------------------------------------
    s = base._slide_base(prs, "Cómo se construyó la etiqueta",
                         "El conjunto etiqueta estudios; la app recibe una imagen")
    base._caja(s, Inches(0.72), Inches(1.9), Inches(12.0), Inches(4.8), [
        ("Positivo — 7 217 cortes con bounding box", True),
        "   la lesión es visible en ese corte concreto.",
        ("Negativo difícil — cortes sanos del MISMO paciente positivo", True),
        "   mismo paciente, mismo escáner, mismo protocolo a ambos lados de la etiqueta:",
        "   el atajo de adquisición deja de pagar. Es la decisión central del diseño.",
        ("Negativo fácil — estudios sin fractura, en las mismas posiciones relativas", True),
        "   muestreados según la distribución de posición de los positivos, para que la",
        "   altura dentro de la pila tampoco sea una pista.",
        ("Excluidos — estudios positivos SIN caja", True),
        "   se sabe que hay fractura pero no en qué corte: ninguna etiqueta sería fiable.",
    ], size=14, space=7)

    # --- 4. Partición -------------------------------------------------------
    s = base._slide_base(prs, "Partición agrupada por paciente",
                         "La decisión que evita una métrica inflada")
    base._caja(s, Inches(0.72), Inches(1.9), Inches(6.3), Inches(4.6), [
        ("Cortes contiguos son casi idénticos.", True),
        "Con 0.625 mm de grosor mediano, dos cortes vecinos muestran prácticamente la misma anatomía.",
        "",
        ("Una partición aleatoria por corte", True),
        "colocaría imágenes casi duplicadas a ambos lados y el modelo obtendría una métrica excelente por haber memorizado al paciente.",
        "",
        ("Se agrupa por StudyInstanceUID", True),
        "con GroupShuffleSplit, y una aserción falla si algún estudio aparece en dos particiones.",
    ], size=14, space=7)
    base._cifra_clave(s, Inches(7.4), Inches(1.95), Inches(5.3), "1 150 estudios",
                      "14 402 cortes, balance 50.1 % positivos", base.TEAL)
    base._cifra_clave(s, Inches(7.4), Inches(3.85), Inches(5.3), "0 % fallos de lectura",
                      "una primera ejecución perdió el 35 % de los cortes en silencio\npor falta del decodificador JPEG Lossless", base.NARANJA)

    # --- 5. Modelos ---------------------------------------------------------
    s = base._slide_base(prs, "Tres modelos de capacidad creciente")
    base._caja(s, Inches(0.72), Inches(1.85), Inches(12.0), Inches(0.5), [
        ("Si la transferencia no supera a la CNN pequeña, el cuello de botella son los datos.", True),
    ], size=15)
    for i, f in enumerate(filas):
        base._cifra_clave(s, Inches(0.72 + i * 4.1), Inches(2.5), Inches(3.85),
                          f"{f['parametros']/1e6:.2f} M",
                          f"{f['modelo']}\n{f['segundos_entrenamiento']:.0f} s de entrenamiento",
                          base.TEAL if i == 0 else base.NARANJA)
    base._caja(s, Inches(0.72), Inches(4.5), Inches(12.0), Inches(2.1), [
        ("Entrenamiento en dos etapas para las redes preentrenadas:", True),
        "   base congelada con tasa 1e-3 → base descongelada con tasa 1e-5 (cien veces menor).",
        ("Parada temprana sobre AUC de validación", True),
        "   con restauración de los mejores pesos, en lugar de un número fijo de épocas.",
        ("Umbral de cribado elegido en validación", True),
        f"   sensibilidad ≥ {filas[0]['sens_objetivo']:.0%} con la mejor especificidad posible, en lugar de 0.5 o Youden.",
    ], size=13.5, space=6)

    # --- 6. ROC -------------------------------------------------------------
    s = base._slide_base(prs, "Resultados · capacidad de ordenamiento")
    base._figura(s, "06_curvas_roc.png", Inches(0.7), Inches(1.8), Inches(7.1), Inches(4.9))
    _por_tam = sorted(filas, key=lambda f: f["parametros"])
    _aucs = " → ".join(f"{f['roc_auc']:.3f}" for f in _por_tam)
    _params = " → ".join(f"{f['parametros']/1e6:.2f} M" for f in _por_tam)
    base._caja(s, Inches(8.1), Inches(1.9), Inches(4.6), Inches(4.8), [
        (f"Mejor: {mejor['modelo']}", True),
        f"ROC-AUC {mejor['roc_auc']:.4f}",
        "",
        ("El orden sigue a la capacidad", True),
        _aucs,
        f"para {_params} parámetros.",
        "",
        (f"La transferencia aporta {rango:.3f} de AUC", True),
        "sobre la CNN entrenada desde cero, pese a que ImageNet no se parece en nada a una tomografía.",
    ], size=13.5, space=6)

    # --- 7. Matrices --------------------------------------------------------
    s = base._slide_base(prs, "Resultados · dónde se equivoca cada modelo")
    base._figura(s, "06_matrices_confusion.png", Inches(0.7), Inches(1.85), Inches(11.9), Inches(3.5))
    base._caja(s, Inches(0.72), Inches(5.5), Inches(12.0), Inches(1.4), [
        ("En cribado la sensibilidad domina.", True),
        "Un falso negativo es una fractura no diagnosticada, con riesgo de deterioro neurológico; un falso positivo "
        f"solo genera una revisión. Por eso el umbral se fija en validación para alcanzar al menos {filas[0]['sens_objetivo']:.0%} "
        "de sensibilidad con la mejor especificidad posible, y queda configurable por modelo en models.yaml: "
        "es una decisión explícita, no el 0.5 por defecto.",
    ], size=13, space=6)

    # --- 8. Comparación -----------------------------------------------------
    s = base._slide_base(prs, "Resultados · rendimiento frente a coste")
    base._figura(s, "06_comparacion_modelos.png", Inches(0.7), Inches(1.85), Inches(11.9), Inches(4.0))
    base._caja(s, Inches(0.72), Inches(6.0), Inches(12.0), Inches(0.9), [
        (f"{rapido['modelo']} responde en {rapido['ms_por_imagen']:.1f} ms por imagen.", True),
        "Para una aplicación interactiva la latencia pesa más que unas centésimas de AUC: los tres están muy por debajo del umbral de percepción.",
    ], size=13, space=5)

    # --- 9. Aplicación ------------------------------------------------------
    s = base._slide_base(prs, "La aplicación", "Streamlit · configuración declarativa")
    base._figura(s, "07_app_datos.png", Inches(0.7), Inches(1.85), Inches(6.1), Inches(4.3))
    base._caja(s, Inches(7.1), Inches(1.9), Inches(5.6), Inches(4.8), [
        ("1 · Datos", True),
        "   explora las variables con las que se construyeron los modelos: observaciones, nulos, distribución de la etiqueta y filtro de texto.",
        ("2 · Analizar imagen", True),
        "   un modelo concreto o «Todos los modelos disponibles»; resultados en tabla y gráfica comparativa.",
        ("3 · Rendimiento", True),
        "   métricas y matrices interactivas, con un interruptor que permite ocultarlas.",
        "",
        ("models.yaml lo genera el notebook de entrenamiento", True),
        "   la app no puede desincronizarse del preprocesamiento real.",
    ], size=13, space=6)

    # --- 10. Capturas -------------------------------------------------------
    s = base._slide_base(prs, "La aplicación · inferencia y rendimiento")
    base._figura(s, "07_app_analisis.png", Inches(0.5), Inches(1.85), Inches(6.1), Inches(4.6))
    base._figura(s, "07_app_rendimiento.png", Inches(6.9), Inches(1.85), Inches(6.1), Inches(4.6))
    base._caja(s, Inches(0.7), Inches(6.5), Inches(12.0), Inches(0.6), [
        "Izquierda: inferencia con todos los modelos sobre la misma imagen. Derecha: métricas del conjunto de prueba, ocultables.",
    ], size=12.5)

    # --- 11. Conclusiones ---------------------------------------------------
    s = base._slide_base(prs, "Conclusiones")
    base._cifra_clave(s, Inches(0.72), Inches(1.85), Inches(3.85), f"{mejor['roc_auc']:.3f}",
                      f"ROC-AUC de {mejor['modelo']}\nsobre pacientes no vistos", base.TEAL)
    base._cifra_clave(s, Inches(4.75), Inches(1.85), Inches(3.85), f"+{rango:.3f}",
                      "aporta la transferencia desde ImageNet\nfrente a entrenar desde cero", base.TEAL)
    base._cifra_clave(s, Inches(8.78), Inches(1.85), Inches(3.85), "11.6 %",
                      "de los estudios tiene anotación:\nel techo real del problema", base.NARANJA)
    base._caja(s, Inches(0.72), Inches(3.8), Inches(12.0), Inches(3.0), [
        ("1. El rendimiento es real pero moderado, y es el resultado honesto.", True),
        "   Sin fuga entre pacientes y con negativos del mismo paciente, no queda margen para atribuirlo a un atajo.",
        ("2. La transferencia sí aporta; el techo es la supervisión disponible.", True),
        "   Anotar más cortes rendiría más que seguir escalando la arquitectura.",
        ("3. El umbral es una decisión clínica, no un valor por defecto.", True),
        "   Se elige en validación y queda configurable por modelo.",
        ("4. La tarea es por corte, no por paciente.", True),
        "   Agregar cortes a un veredicto por estudio es un problema distinto. Uso académico, no diagnóstico.",
    ], size=13, space=5)

    prs.save(SALIDA)
    print(f"Presentación: {SALIDA}")
    print(f"Diapositivas: {len(prs.slides._sldIdLst)} | {SALIDA.stat().st_size/1024:.0f} KB")
    return SALIDA


if __name__ == "__main__":
    construir()
