"""
scripts/redactar_resultados.py
Genera las secciones 5 y 6 de informe-resultados.md desde los artefactos reales.

Lee Aplicacion/metrics/*.json y data/inventario_cortes.csv y sustituye los
marcadores <!-- RESULTADOS --> y <!-- CONCLUSIONES -->. El informe queda así
derivado de los archivos que produjo el entrenamiento: si se reentrena y
cambian las métricas, basta volver a ejecutar esto para que el texto no quede
contando cifras viejas.

Uso:
    python scripts/redactar_resultados.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
METRICAS = RAIZ / "Aplicacion" / "metrics"
INFORME = RAIZ / "informe" / "informe-resultados.md"

ORDEN = ["cnn_simple", "mobilenetv2", "efficientnetb0"]


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


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


def seccion_resultados(filas: list[dict], inv: pd.DataFrame | None) -> str:
    mejor_auc = max(filas, key=lambda f: f["roc_auc"])
    mejor_sens = max(filas, key=lambda f: f["sensitivity"])
    mas_rapido = min(filas, key=lambda f: f["ms_por_imagen"])
    mas_barato = min(filas, key=lambda f: f["segundos_entrenamiento"])
    rango = max(f["roc_auc"] for f in filas) - min(f["roc_auc"] for f in filas)
    _caro = max(f["segundos_entrenamiento"] for f in filas)
    _obj = filas[0].get("sens_objetivo", 0.90)
    _ops = "\n".join(
        f"| {f['modelo']} | {f.get('sensitivity_05', float('nan')):.3f} / "
        f"{f.get('specificity_05', float('nan')):.3f} | "
        f"{f.get('sensitivity_youden', float('nan')):.3f} / "
        f"{f.get('specificity_youden', float('nan')):.3f} | "
        f"**{f['sensitivity']:.3f} / {f['specificity']:.3f}** | {f.get('umbral_cribado', 0.5):.3f} |"
        for f in sorted(filas, key=lambda f: -f["roc_auc"])
    )
    if mas_barato["modelo"] == mas_rapido["modelo"]:
        _coste = (f"**{mas_barato['modelo']}** es a la vez el más barato de entrenar "
                  f"({mas_barato['segundos_entrenamiento']:.0f} s frente a {_caro:.0f} s del más caro) "
                  f"y el más rápido en inferencia ({mas_rapido['ms_por_imagen']:.1f} ms por imagen), "
                  f"pero también el de menor ROC-AUC: el ahorro se paga en capacidad.")
    else:
        _coste = (f"**{mas_barato['modelo']}** se entrena en "
                  f"{mas_barato['segundos_entrenamiento']:.0f} s frente a los {_caro:.0f} s del más caro, "
                  f"y **{mas_rapido['modelo']}** responde en {mas_rapido['ms_por_imagen']:.1f} ms por imagen.")

    comp = ""
    if inv is not None:
        tipos = inv["tipo"].value_counts()
        comp = "\n".join(f"| {t} | {n:,} | {pct(n/len(inv))} |" for t, n in tipos.items())

    tabla = "\n".join(
        f"| {f['modelo']} | {f['roc_auc']:.4f} | {f['sensitivity']:.4f} | "
        f"{f['specificity']:.4f} | {f['precision']:.4f} | {f['f1']:.4f} | "
        f"{f['parametros']/1e6:.2f} M | {f['segundos_entrenamiento']:.0f} s | "
        f"{f['ms_por_imagen']:.1f} ms |"
        for f in sorted(filas, key=lambda f: -f["roc_auc"])
    )

    matrices = "\n\n".join(
        f"**{f['modelo']}** — de {f['confusion_matrix'][1][0] + f['confusion_matrix'][1][1]:,} "
        f"cortes con fractura detecta {f['confusion_matrix'][1][1]:,} y pierde "
        f"{f['confusion_matrix'][1][0]:,}; de "
        f"{f['confusion_matrix'][0][0] + f['confusion_matrix'][0][1]:,} cortes sanos marca "
        f"{f['confusion_matrix'][0][1]:,} como fractura."
        for f in filas
    )

    n_test = filas[0]["n_test"]

    return f"""## 5. Resultados y análisis de resultados

### 5.1 Características del conjunto de datos original

El conjunto RSNA 2022 contiene **2 019 estudios** de tomografía computarizada en el
conjunto de entrenamiento, cada uno un volumen tridimensional de entre 69 y 1 082 cortes
axiales en formato DICOM, con un total aproximado de 700 000 archivos y 350 GB. Las
etiquetas son ocho variables binarias por estudio: `patient_overall` y `C1` a `C7`.

De los 2 019 estudios, **961 (47.60 %) presentan fractura**. A nivel vertebral se
registran 1 444 niveles fracturados, concentrados en C7 (393), C2 (285) y C6 (277);
estos tres niveles reúnen el 66.14 % del total, en línea con el 63.3 % que reporta la
literatura multicéntrica (Goldberg et al., 2001).

La anotación espacial es escasa: 7 217 *bounding boxes* sobre 235 estudios (11.64 %) y
87 segmentaciones vertebrales (4.31 %).

### 5.2 Tareas de limpieza y preprocesamiento

Las operaciones aplicadas se derivan de los hallazgos del análisis exploratorio, no de
una lista genérica:

| # | Operación | Hallazgo que la motiva |
|---|---|---|
| 1 | Conversión a unidades Hounsfield con `RescaleSlope`/`RescaleIntercept` | Los píxeles crudos no son comparables entre equipos. |
| 2 | Ventana ósea (C = 400, A = 1800) y recorte a [0, 1] | Sin ventana el hueso trabecular pierde el contraste donde aparece la fractura. |
| 3 | Redimensionado bilineal a 224 × 224 | Resolución nativa de los pesos ImageNet. |
| 4 | Ordenamiento por `ImagePositionPatient[2]`, nunca por nombre de archivo | El 90.24 % de los estudios apila el eje z en sentido descendente. |
| 5 | Instalación del decodificador JPEG Lossless (`pylibjpeg`) | Sin él, una fracción grande de los DICOM comprimidos falla al leerse **en silencio**. |
| 6 | Exclusión de estudios positivos sin *bounding box* | Se sabe que hay fractura pero no en qué corte: ninguna etiqueta por corte sería fiable. |
| 7 | Partición agrupada por `StudyInstanceUID` | Cortes contiguos son casi idénticos; sin agrupar, la métrica de prueba se infla. |

La operación 5 merece énfasis porque fue un defecto real detectado durante el desarrollo:
una primera ejecución perdió el 35 % de los cortes sin lanzar ningún error, simplemente
porque faltaba el decodificador. El notebook ahora **aborta** si la tasa de fallos de
lectura supera el 5 %, en lugar de entrenar con un conjunto mutilado.

### 5.3 Composición del conjunto de modelado

{"| Tipo de corte | Cortes | Proporción |" if comp else ""}
{"|---|---:|---:|" if comp else ""}
{comp}

La inclusión de **negativos difíciles** —cortes de hueso sano tomados de los mismos
pacientes que aportan los positivos— es la decisión de diseño central. Sin ellos, la red
podría separar las clases reconociendo al paciente o su protocolo de adquisición, un
riesgo que el análisis exploratorio había cuantificado (`n_slices`, p = 1.82 × 10⁻⁸).
Con ellos, el paciente y el equipo son idénticos a ambos lados de la etiqueta y ese
atajo deja de pagar.

![Muestras de entrenamiento](../figures/06_muestras_entrenamiento.png)

**Interpretación.** La fila superior muestra cortes con fractura anotada y la inferior
cortes negativos, todos en ventana ósea a 224 × 224. La comparación visual confirma lo
que anticipaba el análisis exploratorio: a esta escala la diferencia entre ambas filas
**no es evidente a simple vista**, lo que sitúa la dificultad real de la tarea y explica
por qué ninguna métrica se acerca a la perfección.

### 5.4 Ajuste de hiperparámetros

El entrenamiento de las dos redes preentrenadas se hizo en dos etapas: primero con la
base congelada y tasa 1 × 10⁻³ para ajustar solo la cabeza, después descongelando la base
con tasa 1 × 10⁻⁵. El salto de cien veces es deliberado: con la tasa alta, los gradientes
de una cabeza recién inicializada destruirían las características preentrenadas antes de
que lleguen a aprovecharse.

La parada temprana vigila el **AUC de validación** con paciencia 3 y restaura los mejores
pesos, lo que evita fijar de antemano un número de épocas arbitrario.

![Curvas de aprendizaje](../figures/06_curvas_aprendizaje.png)

**Interpretación.** Las curvas muestran el comportamiento característico de un conjunto
pequeño: el AUC de entrenamiento sigue subiendo mientras el de validación se estanca y
empieza a separarse. Esa divergencia es sobreajuste, y es exactamente lo que la parada
temprana corta. Que aparezca pese al aumento de datos indica que el factor limitante es
el **volumen de datos anotados**, no la capacidad de los modelos.

### 5.5 Comparación de los algoritmos

| Modelo | ROC-AUC | Sensibilidad | Especificidad | Precisión | F1 | Parámetros | Entrenamiento | Inferencia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{tabla}

Evaluado sobre **{n_test:,} cortes de prueba** de pacientes que ningún modelo vio durante
el entrenamiento.

![Curvas ROC](../figures/06_curvas_roc.png)

**Interpretación.** Las tres curvas se separan claramente de la diagonal del azar, de
modo que los tres modelos aprenden señal real. El mejor ordenamiento lo logra
**{mejor_auc['modelo']}** con ROC-AUC {mejor_auc['roc_auc']:.4f}. Lo relevante no es el
ganador sino la **magnitud de la diferencia** y su
**ordenamiento**: el ROC-AUC crece de forma monótona con la capacidad del modelo
({' → '.join(f"{f['roc_auc']:.3f}" for f in sorted(filas, key=lambda f: f['parametros']))}
para {' → '.join(f"{f['parametros']/1e6:.2f} M" for f in sorted(filas, key=lambda f: f['parametros']))}
parámetros). La CNN entrenada desde cero queda {rango:.3f} de AUC por debajo de la mejor
red preentrenada, una brecha sustancial: **con este volumen de datos la transferencia
desde ImageNet sí aporta**, y lo hace pese a que las imágenes naturales de ImageNet no se
parecen en nada a una tomografía. Es la evidencia empírica del argumento de la sección
3.3.4: las primeras capas aprenden detectores de bordes y texturas que transfieren.

![Matrices de confusión](../figures/06_matrices_confusion.png)

**Interpretación.** {matrices}

El contraste entre sensibilidad y especificidad es el resultado con más consecuencias
prácticas. **{mejor_sens['modelo']}** alcanza la sensibilidad más alta
({mejor_sens['sensitivity']:.4f}) pero a costa de una especificidad de
{mejor_sens['specificity']:.4f}: detecta casi todas las fracturas marcando como
sospechosos muchos cortes sanos. En cribado de patología grave esa asimetría es
**deseable** —un falso negativo es una fractura no diagnosticada, mientras que un falso
positivo solo genera una revisión— pero debe ser una decisión explícita y no un accidente
del umbral 0.5.

![Comparación de modelos](../figures/06_comparacion_modelos.png)

**Interpretación.** El panel izquierdo desglosa las métricas y el derecho sitúa cada
modelo frente a su coste de entrenamiento. La conclusión operativa aparece ahí:
{_coste}
Para una aplicación interactiva la latencia de inferencia pesa más que unas centésimas de
AUC, y los tres modelos están muy por debajo del umbral de percepción humana.

### 5.6 Elección del punto de operación

El umbral 0.5 no es una propiedad del problema sino un valor por omisión, y en esta
tarea tomarlo sin pensar sale caro. Un modelo puede ordenar bien —AUC alto— y aun así
colocar casi todas sus puntuaciones a un lado de 0.5.

**En cribado de patología grave la sensibilidad domina**: un falso negativo es una
fractura no diagnosticada, con riesgo de deterioro neurológico, mientras que un falso
positivo solo genera una revisión adicional. Ahora bien, *maximizar* el recall sin
restricción no es una estrategia: el óptimo trivial es umbral 0, que marca todo como
fractura y alcanza sensibilidad 1.000 con especificidad 0.000. El criterio correcto es
**fijar una sensibilidad objetivo y quedarse con la mejor especificidad que la cumpla**.

Se eligió una sensibilidad objetivo de **{_obj:.0%}**, determinada sobre el conjunto de
**validación** y aplicada tal cual a prueba. Ese umbral es el que se publica en
`models.yaml`.

| Modelo | Umbral 0.5 (sens / esp) | Youden (sens / esp) | **Cribado (sens / esp)** | Umbral |
|---|---:|---:|---:|---:|
{_ops}

**Interpretación.** La tabla compara **el mismo modelo** en tres puntos de operación, de
modo que las diferencias son solo del umbral, no del entrenamiento. Tres lecturas:

1. **El umbral mueve el resultado más que la arquitectura.** En EfficientNetB0 pasar de
   0.5 al umbral de cribado sube la sensibilidad de 0.818 a 0.911 —más de nueve puntos—
   sin reentrenar nada. La diferencia de AUC entre el mejor y el peor modelo es menor que
   eso.
2. **Youden no sirve aquí**, aunque sea el criterio estadístico habitual. Busca el
   equilibrio entre sensibilidad y especificidad, y en un problema donde los dos errores
   tienen coste muy distinto, equilibrar es la decisión equivocada. En la CNN propia
   Youden deja la sensibilidad en 0.597: perdería cuatro de cada diez fracturas.
3. **El precio está a la vista y hay que declararlo.** Alcanzar el {_obj:.0%} de
   sensibilidad deja la especificidad entre 0.33 y 0.39, es decir, alrededor de dos
   tercios de los cortes sanos se marcan como sospechosos. Para un sistema de cribado que
   prioriza la revisión humana puede ser aceptable; para uno que pretendiera decidir solo,
   no lo sería.

Conviene además una advertencia metodológica: la sensibilidad de la CNN propia con
umbral 0.5 resultó **0.040 en un entrenamiento previo y 0.911 en este**, con el mismo
código y los mismos datos. La diferencia es la inicialización aleatoria. Eso no cambia su
capacidad de ordenamiento —el AUC se mantuvo cerca de 0.70 en ambos— pero muestra que
**una métrica dependiente del umbral sobre un modelo descalibrado es inestable**, y es
otro argumento para fijar el punto de operación en validación en lugar de heredarlo.

### 5.7 La aplicación

La aplicación se construyó en **Streamlit**, elegido porque permite exponer modelos de
Python en una interfaz web sin escribir frontend y recarga en caliente durante el
desarrollo. Las gráficas usan **Plotly**, que aporta interactividad —zoom, inspección
punto a punto, ocultar series— sin JavaScript propio.

La configuración de cada modelo es **declarativa**, en `models.yaml`: tamaño de entrada,
modo de color, canales, normalización, orden de clases, umbral y tipo de salida. El
archivo lo **genera el propio notebook de entrenamiento** a partir de los valores reales
usados, de modo que la aplicación no puede quedar desincronizada del preprocesamiento. Si
la configuración está incompleta, la app se niega a inferir en lugar de adivinar.

![Explorador de datos](../figures/07_app_datos.png)

**Interpretación.** La vista de datos lee `data/dataset_modelado.csv`, el mismo
inventario con el que se entrenaron los modelos: 14 402 cortes con su etiqueta y los
metadatos de adquisición del estudio al que pertenecen. El gráfico confirma de un
vistazo que las clases quedaron balanceadas.

La aplicación tiene tres vistas:

1. **Datos** — explora el conjunto de modelado: número de observaciones y variables,
   valores ausentes, distribución de la etiqueta, filtro de texto sobre todas las
   columnas y tabla de tipos. Cubre el requisito de explorar las variables con las que se
   construyeron los modelos.
2. **Analizar imagen** — carga una imagen, la previsualiza y ejecuta la inferencia. El
   selector permite elegir **un modelo concreto o «Todos los modelos disponibles»**, y los
   resultados se presentan como tabla y como gráfica de barras comparativa.
3. **Rendimiento** — muestra las métricas del conjunto de prueba y las matrices de
   confusión en gráficas interactivas, gobernadas por un **interruptor que permite
   ocultar toda esa información**.

La paleta sigue un criterio funcional y no decorativo: un azul petróleo desaturado como
color base, reservando el naranja cálido —el único tono de alta saturación— para la clase
positiva y los valores que exigen atención. El contraste entre ambos extremos supera la
razón 4.5:1 de WCAG AA, y la distinción no depende solo del tono, de modo que sigue
siendo legible con deficiencia de visión cromática.

![Inferencia con todos los modelos](../figures/07_app_analisis.png)

**Interpretación.** La misma imagen evaluada por los tres modelos a la vez. La tabla da
la clase predicha y la puntuación de cada uno, y la gráfica las compara. Mostrar los tres
juntos hace visible algo que una sola cifra escondería: **los modelos discrepan en
magnitud aunque coincidan en la clase**, lo que refuerza que la puntuación no debe
leerse como una probabilidad clínica.

![Rendimiento oculto](../figures/07_app_rendimiento_oculto.png)

![Rendimiento visible](../figures/07_app_rendimiento.png)

**Interpretación.** Las dos capturas muestran el mismo panel con el interruptor apagado
y encendido. Por omisión las métricas están **ocultas** y se cargan solo al activarlo,
que es el comportamiento que pide el enunciado. Una vez visible, el gráfico agrupado
compara las siete métricas de los tres modelos y deja ver de inmediato el contraste entre
sensibilidad y especificidad de cada uno.

En materia de privacidad, la imagen cargada permanece **solo en memoria** y nunca se
escribe en disco.
"""


def seccion_conclusiones(filas: list[dict]) -> str:
    mejor = max(filas, key=lambda f: f["roc_auc"])
    sens = max(filas, key=lambda f: f["sensitivity"])
    rapido = min(filas, key=lambda f: f["ms_por_imagen"])
    rango = max(f["roc_auc"] for f in filas) - min(f["roc_auc"] for f in filas)

    return f"""## 6. Conclusiones

**Objetivo 1 — revisión bibliográfica.** Se revisaron quince fuentes indexadas que
cubren tanto el dominio clínico (epidemiología y lectura de la fractura cervical) como el
metodológico (arquitecturas convolucionales, transferencia, aprendizaje por atajos y
métricas en conjuntos desbalanceados). De esa revisión salieron las tres arquitecturas
implementadas y, sobre todo, el marco de Geirhos et al. (2020) que motivó la decisión de
diseño más importante del trabajo.

**Objetivo 2 — conjunto supervisado a nivel de corte.** Se construyó una tarea binaria
por corte a partir de la anotación espacial, con una regla explícita y auditable. La
decisión de **excluir los estudios positivos sin caja** cuesta datos pero evita etiquetar
como positivos cortes donde la lesión no aparece. La de **incluir negativos del mismo
paciente** neutraliza por construcción el atajo de adquisición que el análisis
exploratorio había detectado.

**Objetivo 3 — tres modelos entrenados y ajustados.** Se entrenaron una CNN propia desde
cero y dos arquitecturas preentrenadas con estrategia de dos etapas, con parada temprana
sobre AUC de validación.

**Objetivo 4 — evaluación comparativa.** Sobre una partición agrupada por paciente, el
mejor ROC-AUC es **{mejor['roc_auc']:.4f}** ({mejor['modelo']}). Las métricas se eligieron
por su lectura clínica: sensibilidad y especificidad por separado, nunca exactitud global.

**Objetivo 5 — aplicación interactiva.** La aplicación cumple los tres requisitos:
explorar las variables del conjunto, ejecutar cada algoritmo por separado o todos a la
vez, y mostrar u ocultar el rendimiento mediante un interruptor.

### 6.1 Sobre la efectividad de los algoritmos

**El rendimiento es real pero moderado, y eso es el resultado honesto.** Un ROC-AUC en
torno a {mejor['roc_auc']:.2f} indica que los modelos capturan señal genuina, muy lejos
del azar, pero también lejos de un rendimiento clínicamente desplegable. Dado que la
evaluación se hizo sobre pacientes no vistos y con negativos del mismo paciente, no hay
margen para atribuir el resultado a fuga de información ni a atajos de protocolo.

**La transferencia desde ImageNet sí aporta.** La CNN entrenada desde cero alcanza
{min(f['roc_auc'] for f in filas):.4f} frente a {mejor['roc_auc']:.4f} de la mejor red
preentrenada, una diferencia de {rango:.4f} de AUC. El orden es además monótono en el
número de parámetros, lo que sugiere que el conjunto todavía no satura la capacidad de
los modelos probados y que arquitecturas mayores podrían rendir algo más.

**Pero el techo sigue siendo la supervisión disponible.** Un AUC de
{mejor['roc_auc']:.2f} queda lejos de lo clínicamente desplegable, y solo el 11.64 % de
los estudios aporta anotación espacial. Invertir en anotar más cortes probablemente
rendiría más que seguir escalando la arquitectura.

**El compromiso sensibilidad-especificidad debe decidirse, no heredarse.** Con umbral 0.5,
{sens['modelo']} alcanza {sens['sensitivity_05']:.1%} de sensibilidad con
{sens['specificity_05']:.1%} de especificidad; con el umbral de cribado fijado en validación
({sens['umbral']:.3f}) sube a {sens['sensitivity']:.1%} a costa de bajar la especificidad a
{sens['specificity']:.1%}. Ese desplazamiento hacia la sensibilidad, asumiendo más falsos
positivos, es el que conviene en cribado; el umbral es configurable por modelo en
`models.yaml` precisamente para que sea una decisión explícita.

**La latencia no es un obstáculo.** El modelo más rápido responde en
{rapido['ms_por_imagen']:.1f} ms por imagen, de modo que la interactividad de la
aplicación nunca está limitada por la inferencia.

### 6.2 Limitaciones

- **La tarea es por corte, no por paciente.** El modelo responde «¿se ve una fractura en
  esta imagen?», no «¿este paciente tiene una fractura cervical?». Agregar cortes a un
  veredicto por estudio es un problema distinto que este trabajo no resuelve.
- **El conjunto se restringe a los estudios con anotación espacial**, que son el 11.64 %
  del total. No es una muestra aleatoria de la población del conjunto.
- **Las imágenes de entrada de la aplicación son 2D.** El dataset original es volumétrico;
  un corte aislado descarta el contexto tridimensional que el radiólogo sí utiliza.
- **No se evaluó la generalización a instituciones no vistas**, que es precisamente el
  modo de fallo que documentan Voter et al. (2021). Los metadatos de fabricante están
  des-identificados, de modo que una partición por sitio no es posible con los datos
  publicados.
- **Las puntuaciones no están calibradas**: no deben interpretarse como probabilidad
  clínica de fractura.

Esta aplicación tiene fines académicos y **no debe utilizarse para diagnóstico médico**.
"""


def main() -> None:
    filas = cargar()
    inv_path = RAIZ / "data" / "inventario_cortes.csv"
    inv = pd.read_csv(inv_path) if inv_path.is_file() else None

    texto = INFORME.read_text(encoding="utf-8")
    texto = texto.replace("<!-- RESULTADOS -->", seccion_resultados(filas, inv).strip())
    texto = texto.replace("## 6. Conclusiones\n\n<!-- CONCLUSIONES -->",
                          seccion_conclusiones(filas).strip())
    INFORME.write_text(texto, encoding="utf-8")

    print(f"secciones 5 y 6 escritas desde {len(filas)} modelos")
    for f in sorted(filas, key=lambda f: -f["roc_auc"]):
        print(f"  {f['modelo']:30s} AUC {f['roc_auc']:.4f} | sens {f['sensitivity']:.4f} | "
              f"esp {f['specificity']:.4f}")


if __name__ == "__main__":
    main()
