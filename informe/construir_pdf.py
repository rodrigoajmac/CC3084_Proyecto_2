"""
informe/construir_pdf.py
Convierte informe.md en informe.pdf (Miembro D).

Uso:
    python informe/construir_pdf.py

No requiere pandoc ni LaTeX. Intenta dos motores de render, en este orden:

1. **WeasyPrint** — motor preferido. En Windows necesita las librerías GTK
   (Pango/Cairo), que no vienen instaladas por defecto.
2. **Edge o Chrome en modo headless** — está en cualquier Windows 11 y no
   necesita instalar nada. Las versiones actuales de Chromium sí respetan las
   cajas de margen de `@page`, de modo que el PDF conserva la numeración de
   páginas y el pie de página.

Verificado con Edge headless: 20 páginas, las 15 figuras incrustadas,
numeración al pie.

Las rutas de las imágenes se resuelven relativas a la carpeta `informe/`, así
que `../figures/x.png` funciona tal como está escrito en el markdown.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

AQUI = Path(__file__).resolve().parent
ENTRADA = AQUI / "informe.md"
SALIDA = AQUI / "informe.pdf"

CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm 2cm 2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 8.5pt;
        color: #6b7280;
    }
    @bottom-right {
        content: "CC3084 · Proyecto 2 · Reto 20";
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 7.5pt;
        color: #9ca3af;
    }
}
@page :first {
    @bottom-right { content: ""; }
}

body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #1f2933;
    hyphens: auto;
    text-align: justify;
}

h1 {
    font-size: 20pt;
    color: #264653;
    border-bottom: 3px solid #2a9d8f;
    padding-bottom: 0.35em;
    margin: 0 0 0.8em 0;
    line-height: 1.2;
    text-align: left;
}
h2 {
    font-size: 14pt;
    color: #264653;
    border-bottom: 1px solid #cbd5dd;
    padding-bottom: 0.22em;
    margin-top: 1.6em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
    text-align: left;
}
h3 {
    font-size: 11.5pt;
    color: #2a9d8f;
    margin-top: 1.25em;
    margin-bottom: 0.45em;
    page-break-after: avoid;
    text-align: left;
}
h4 { font-size: 10.5pt; color: #457b9d; page-break-after: avoid; text-align: left; }

p { margin: 0.5em 0; orphans: 3; widows: 3; }

/* Tablas */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.9em 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}
thead { background: #264653; color: #ffffff; }
th, td {
    border: 1px solid #cbd5dd;
    padding: 4.5px 7px;
    text-align: left;
    vertical-align: top;
}
th { font-weight: 600; }
tbody tr:nth-child(even) { background: #f4f6f7; }

/* Figuras */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.9em auto 0.4em auto;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    page-break-inside: avoid;
}

/* Citas: se usan para advertencias y notas de coordinación */
blockquote {
    border-left: 4px solid #e76f51;
    background: #fdf6f3;
    margin: 1em 0;
    padding: 0.55em 1em;
    color: #4a3b36;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
blockquote p { margin: 0.3em 0; }

code {
    font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
    background: #eef2f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.6pt;
    color: #9d174d;
}
pre {
    background: #f7f9fa;
    border: 1px solid #dde3e8;
    border-left: 3px solid #2a9d8f;
    border-radius: 3px;
    padding: 0.7em 0.9em;
    font-size: 8pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; color: #1f2933; }

hr {
    border: none;
    border-top: 1px solid #d7dee3;
    margin: 1.6em 0;
}

ul, ol { margin: 0.45em 0; padding-left: 1.5em; }
li { margin: 0.28em 0; }

strong { color: #16323f; }
a { color: #2a6f97; text-decoration: none; }
"""


def _render_weasyprint(html: str) -> bool:
    """Motor preferido: respeta la numeración de páginas de @page."""
    try:
        from weasyprint import CSS as WeasyCSS, HTML
    except Exception as exc:
        print(f"[1/2] WeasyPrint no disponible ({type(exc).__name__}); se usará el navegador.")
        return False
    HTML(string=html, base_url=str(AQUI)).write_pdf(SALIDA, stylesheets=[WeasyCSS(string=CSS)])
    print("[1/2] Render con WeasyPrint (con numeración de páginas).")
    return True


def _buscar_navegador() -> str | None:
    """Localiza Edge o Chrome para el render de respaldo."""
    for nombre in ("msedge", "chrome", "chromium", "google-chrome"):
        ruta = shutil.which(nombre)
        if ruta:
            return ruta
    programas = [os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                 os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")]
    candidatos = [
        Path(base) / sub
        for base in programas
        for sub in (r"Microsoft\Edge\Application\msedge.exe",
                    r"Google\Chrome\Application\chrome.exe")
    ]
    return next((str(p) for p in candidatos if p.exists()), None)


def _render_navegador(html: str) -> bool:
    """Respaldo: Edge/Chrome headless. Sin numeración de páginas."""
    navegador = _buscar_navegador()
    if navegador is None:
        print("[2/2] No se encontró Edge ni Chrome.", file=sys.stderr)
        return False

    # El HTML debe quedar dentro de informe/ para que ../figures/ resuelva.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="_informe_", dir=AQUI,
        encoding="utf-8", delete=False,
    )
    try:
        tmp.write(html)
        tmp.close()
        perfil = tempfile.mkdtemp(prefix="_edgeprofile_")
        comando = [
            navegador,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={perfil}",
            "--no-pdf-header-footer",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={SALIDA}",
            Path(tmp.name).as_uri(),
        ]
        proceso = subprocess.run(comando, capture_output=True, text=True, timeout=180)
        shutil.rmtree(perfil, ignore_errors=True)
        if not SALIDA.exists() or SALIDA.stat().st_size == 0:
            print(f"[2/2] El navegador no generó el PDF.\n{proceso.stderr[-800:]}", file=sys.stderr)
            return False
        print(f"[2/2] Render con {Path(navegador).stem} headless.")
        return True
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def construir() -> Path:
    if not ENTRADA.exists():
        raise FileNotFoundError(f"No existe {ENTRADA}")

    texto = ENTRADA.read_text(encoding="utf-8")

    faltantes = []
    for linea in texto.splitlines():
        if linea.strip().startswith("!["):
            ruta = linea.split("](", 1)[-1].rstrip(")").strip()
            if not ruta.startswith(("http://", "https://")) and not (AQUI / ruta).exists():
                faltantes.append(ruta)
    if faltantes:
        print("AVISO: figuras referenciadas que no existen en disco:")
        for ruta in faltantes:
            print(f"  - {ruta}")
        print("  El PDF se generará con esos huecos. Ejecuta los notebooks y exporta "
              "las figuras a figures/ antes de la entrega.\n")

    html_cuerpo = markdown.markdown(
        texto,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    html = (
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
        "<title>Informe — Proyecto 2 · Reto 20</title>"
        f"<style>{CSS}</style></head>"
        f"<body>{html_cuerpo}</body></html>"
    )

    # Se conserva siempre el HTML: sirve de respaldo si ningún motor funciona y
    # permite revisar el maquetado en el navegador sin regenerar el PDF.
    respaldo = AQUI / "informe.html"
    respaldo.write_text(html, encoding="utf-8")

    if not (_render_weasyprint(html) or _render_navegador(html)):
        print(
            "\nERROR: ningún motor de render funcionó.\n"
            f"  Queda el HTML maquetado en {respaldo}: ábrelo en el navegador y "
            "usa Imprimir > Guardar como PDF.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"PDF generado: {SALIDA}  ({SALIDA.stat().st_size / 1024:.0f} KB)")
    return SALIDA


if __name__ == "__main__":
    construir()
