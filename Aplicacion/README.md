# Explorador académico de modelos cervicales

Aplicación Streamlit para cargar y comparar modelos **ya entrenados** de clasificación
binaria. La aplicación no entrena, ajusta ni convierte modelos, no guarda las imágenes
subidas y no debe utilizarse para diagnóstico médico.

> El dataset RSNA 2022 del proyecto contiene tomografías computarizadas axiales en DICOM,
> no radiografías laterales. La entrada 2D de esta aplicación solo debe usarse si el modelo
> fue entrenado y validado para ese tipo exacto de imagen.

## Instalación y ejecución

Desde la carpeta `Aplicacion`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Instala el framework que corresponda al modelo:
python -m pip install tensorflow
# o: python -m pip install torch
streamlit run app.py
```

La interfaz puede iniciarse sin TensorFlow ni PyTorch; esos paquetes se importan solo al
ejecutar un modelo de su framework.

## Obtener los modelos entrenados

Los `.keras` pesan entre 3 y 48 MB y **no se versionan** (`models/` está en
`.gitignore`). Se generan con `notebooks/06-modelado.ipynb`, que se ejecuta en Kaggle
con GPU y el dataset de la competencia adjunto, y produce en `/kaggle/working`:

```
models/cnn_simple.keras  models/mobilenetv2.keras  models/efficientnetb0.keras
metrics/*.json           models.yaml               figures/06_*.png
```

Descarga `models/` y `metrics/` a esta carpeta y `models.yaml` a su raíz. El
`models.yaml` del repositorio ya corresponde a esos tres modelos: **no lo edites a
mano**, lo regenera el notebook con los valores reales del entrenamiento.

Las métricas sí se versionan (son unos pocos KB), de modo que la vista de Rendimiento
funciona incluso antes de descargar los modelos.

## Configurar modelos (para añadir uno distinto)

1. Copia el archivo del modelo a `models/` (esta carpeta está excluida de Git para evitar
   publicar artefactos grandes o sensibles).
2. Copia una entrada apropiada de `models.example.yaml` a `models.yaml`.
3. Sustituye cada valor con la información real del entrenamiento.

Son obligatorios el tamaño de entrada, modo de color, canales, normalización, orden de
clases, índice de fractura, umbral y tipo de salida. La app se niega a inferir cuando la
configuración es incompleta. Keras admite `.keras`, `.h5` y SavedModel. PyTorch admite
modelos exportados como TorchScript (`.pt`/`.pth`); un `state_dict` no puede cargarse sin
el código de su arquitectura y por ello no se acepta como si fuera un modelo completo.

`output_type: probability` significa que el archivo ya devuelve probabilidades.
`output_type: logit` aplica sigmoid a una salida binaria o softmax a una salida multiclase.
Estas puntuaciones no representan riesgo clínico ni garantizan calibración.
Cuando el modelo tiene una sola salida, esta debe corresponder a la clase indicada por
`fracture_index`; con dos salidas, su orden debe coincidir exactamente con `class_order`.

## Metadatos y métricas

- Configura `manifest_path` en `app_config.yaml` para explorar un CSV existente. Si no
  existe, la app no inventa conteos ni variables.
- Para cada modelo, configura `metrics_path` con un JSON o CSV de métricas del conjunto
  de prueba. El JSON puede contener `sensitivity`, `specificity`, `precision`, `recall`,
  `f1`, `roc_auc`, `accuracy` y una matriz `confusion_matrix` de 2 × 2.
- Un CSV puede tener columnas `metric,value`, o una fila con cada métrica como columna.

Ejemplo de JSON:

```json
{
  "sensitivity": 0.0,
  "specificity": 0.0,
  "confusion_matrix": [[0, 0], [0, 0]]
}
```

Los ceros anteriores ilustran el formato, no son resultados del proyecto.

## Grad-CAM

Grad-CAM queda deshabilitado hasta contar con un modelo real, conocer su capa convolucional
objetivo y validar visualmente la implementación contra su arquitectura. Mostrar un mapa
genérico sin esas comprobaciones podría inducir una interpretación anatómica falsa.

## Privacidad y límites

La imagen subida permanece en memoria durante la sesión y no se escribe en disco. Evita
subir archivos con identificadores visibles. La aplicación no comprueba automáticamente
que la anatomía, proyección o protocolo sean compatibles con el modelo.

La carga DICOM no está habilitada en esta interfaz 2D: hacerlo correctamente requiere
conocer la conversión a HU, la ventana y la selección de cortes usadas por cada modelo.
No se adopta una ventana genérica porque alteraría el preprocesamiento de inferencia.
