"""
scripts/reconstruir_notebook_ejecutado.py
Rellena notebooks/06-modelado.ipynb con la salida real de su ejecución en Kaggle.

Por qué existe: el notebook se ejecuta en Kaggle (necesita el dataset de 350 GB
y GPU) y Kaggle **no expone por API el .ipynb ya ejecutado** — `kernels/pull`
devuelve solo el código y el ZIP de salida solo trae los archivos de usuario.
Al commitear el notebook, por tanto, llegaba sin ningún output.

Cómo lo resuelve: cada celda de código imprime `@@CELDA n@@` como primera línea.
Este script descarga el log del kernel, lo parte por ese centinela y adjunta a
cada celda **su** stdout real, más las figuras que esa celda guardó. No se
inventa nada: si una celda no produjo salida, se queda vacía.

Uso:
    python scripts/reconstruir_notebook_ejecutado.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NB = RAIZ / "notebooks" / "06-modelado.ipynb"
FIGURAS = RAIZ / "figures"
USUARIO, SLUG = "juneherrerawatanabe", "cc3084-p2-modelado-cervical"

# Ruido del runtime que no aporta nada a quien lee el notebook.
RUIDO = (
    "Debugger warning", "frozen modules", "PYDEVD", "MissingIDFieldWarning",
    "_validate(", "validate(nb)", "NbConvertApp", "nbformat", "absl::",
    "oneDNN", "tensorflow/core", "WARNING:tensorflow", "external/local",
    "gpu_device.cc", "meta_optimizer", "successful NUMA", "warnings.warn",
    "SyntaxWarning", "UserWarning", "cells[i][c] = re.sub", "text = re.sub",
    "To enable the following instructions", "rebuild TensorFlow",
)


def token() -> str:
    import subprocess
    out = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "kaggle_mcp_token.py"), "--print"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(RAIZ),
    )
    t = (out.stdout or "").strip().splitlines()[-1].strip() if out.stdout else ""
    if len(t) < 20:
        raise SystemExit("no se pudo obtener el token de Kaggle")
    return t


def log_de_kaggle() -> str:
    qs = urllib.parse.urlencode({"userName": USUARIO, "kernelSlug": SLUG})
    req = urllib.request.Request(f"https://www.kaggle.com/api/v1/kernels/output?{qs}")
    req.add_header("Authorization", f"Bearer {token()}")
    datos = json.loads(urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace"))
    partes = []
    for e in json.loads(datos.get("log") or "[]"):
        s = (e.get("data", "") if isinstance(e, dict) else str(e)).rstrip()
        if s and not any(r in s for r in RUIDO):
            partes.append(s)
    return "\n".join(partes)


def figuras_de(fuente: str) -> list[str]:
    """Nombres de PNG que esta celda guarda, en el orden en que los guarda."""
    nombres = re.findall(r'["\']([\w./-]+\.png)["\']', fuente)
    nombres += re.findall(r'guardar\(fig,\s*["\']([\w.-]+)["\']', fuente)
    vistos, salida = set(), []
    for n in nombres:
        base = Path(n).name
        if not base.endswith(".png"):
            base += ".png"
        if base not in vistos and (FIGURAS / base).is_file():
            vistos.add(base)
            salida.append(base)
    return salida


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if not any("@@CELDA" in "".join(c["source"]) for c in codigo):
        raise SystemExit("El notebook no tiene centinelas: regenéralo con nb_train.py.")

    log = log_de_kaggle()
    if "@@CELDA" not in log:
        raise SystemExit(
            "El log de Kaggle no contiene centinelas. ¿La última ejecución es la "
            "versión con centinelas y terminó COMPLETE?"
        )

    # Partir el log por centinela: {n_celda: stdout}
    trozos: dict[int, str] = {}
    actual = None
    acumulado: list[str] = []
    for linea in log.splitlines():
        m = re.match(r"@@CELDA (\d+)@@", linea.strip())
        if m:
            if actual is not None:
                trozos[actual] = "\n".join(acumulado).strip()
            actual, acumulado = int(m.group(1)), []
        elif actual is not None:
            acumulado.append(linea)
    if actual is not None:
        trozos[actual] = "\n".join(acumulado).strip()

    con_texto = con_figura = 0
    for i, celda in enumerate(codigo, start=1):
        fuente = "".join(celda["source"])
        salidas = []

        texto = trozos.get(i, "")
        if texto:
            salidas.append({"output_type": "stream", "name": "stdout",
                            "text": [l + "\n" for l in texto.splitlines()]})
            con_texto += 1

        for png in figuras_de(fuente):
            b64 = base64.b64encode((FIGURAS / png).read_bytes()).decode("ascii")
            salidas.append({"output_type": "display_data",
                            "data": {"image/png": b64, "text/plain": [f"<Figure: {png}>"]},
                            "metadata": {}})
            con_figura += 1

        celda["outputs"] = salidas
        celda["execution_count"] = i

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"notebook: {NB}")
    print(f"celdas de código: {len(codigo)} | con stdout: {con_texto} | figuras incrustadas: {con_figura}")
    print(f"tamaño: {NB.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
