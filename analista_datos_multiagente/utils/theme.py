"""Sistema de diseno visual: CSS de vanguardia + plantilla Plotly oscura.

Inyecta un tema tipo dashboard premium (fondo oscuro, tarjetas "glass", tipografia
Space Grotesk/Inter) via `st.markdown(unsafe_allow_html=True)`. El acento de color
cambia segun el dominio de negocio seleccionado (Parte A), reforzando visualmente
el filtro sin tocar ninguna logica de computo.

Este modulo es PURAMENTE de presentacion: no calcula ni transforma datos.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .domain_config import get_domain

# Paleta base (modo "general", sin dominio de negocio activo).
BG = "#0B0F14"
SURFACE = "rgba(255,255,255,0.04)"
BORDER = "rgba(255,255,255,0.08)"
TEXT_PRIMARY = "#E6E8EB"
TEXT_SECONDARY = "#9AA4B2"
ACCENT_PRIMARY = "#8B5CF6"      # violeta electrico
ACCENT_SECONDARY = "#22D3EE"    # cian
ACCENT_WARN = "#FBBF24"         # amber
ACCENT_DANGER = "#FF6B6B"       # coral

_TEMPLATE_NAME = "analista_dark"


def _register_plotly_template(accent_primary: str, accent_secondary: str) -> None:
    """Registra (o actualiza) la plantilla oscura custom de Plotly."""
    seq = [accent_primary, accent_secondary, ACCENT_WARN, ACCENT_DANGER,
           "#34D399", "#F472B6", "#60A5FA", "#A78BFA"]
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=16,
                             color=TEXT_PRIMARY)),
        colorway=seq,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)",
                  linecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)",
                  linecolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    pio.templates[_TEMPLATE_NAME] = template
    pio.templates.default = _TEMPLATE_NAME


def apply_theme(domain: str | None = None) -> dict:
    """Inyecta el CSS de vanguardia y la plantilla Plotly oscura para `domain`.

    Devuelve la config del dominio (util para que la UI reuse icono/label/color).
    """
    cfg = get_domain(domain)
    accent = cfg["accent_primary"]
    accent2 = cfg["accent_secondary"]

    _register_plotly_template(accent, accent2)

    # Carga de tipografia NO bloqueante ("ir a la segura" con internet lento):
    # en vez de `@import` dentro del <style> (que retiene el render de todo el
    # bloque hasta resolver la peticion externa), usamos el patron loadCSS:
    # el <link> se pide con media="print" (no bloquea) y solo al cargar se
    # conmuta a media="all". Si la red esta lenta o falla, la pagina ya se
    # pinto con las fuentes de sistema (fallback "sans-serif" siempre presente
    # en cada font-family de este archivo), sin quedar nunca en blanco.
    font_loader = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap"
          media="print" onload="this.media='all'">
    <noscript>
      <link rel="stylesheet"
            href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap">
    </noscript>
    """
    st.markdown(font_loader, unsafe_allow_html=True)

    css = f"""
    <style>
    :root {{
        --bg: {BG};
        --surface: {SURFACE};
        --border: {BORDER};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --accent: {accent};
        --accent-2: {accent2};
        --warn: {ACCENT_WARN};
        --danger: {ACCENT_DANGER};
    }}

    .stApp {{
        background: radial-gradient(circle at 15% 0%, rgba(139,92,246,0.10), transparent 40%),
                    radial-gradient(circle at 85% 10%, rgba(34,211,238,0.08), transparent 45%),
                    var(--bg);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }}

    /* --- Header nativo de Streamlit (Share/Estrella/GitHub/Deploy) ---
       Por defecto queda blanco aunque .stApp este oscuro; lo tenimos del mismo
       fondo para que no se vea como una franja rota en la parte superior. */
    header[data-testid="stHeader"] {{
        background: var(--bg);
    }}
    div[data-testid="stToolbar"] {{
        background: transparent;
    }}
    div[data-testid="stToolbar"] button,
    div[data-testid="stToolbar"] svg,
    header[data-testid="stHeader"] svg {{
        color: var(--text-primary) !important;
        fill: var(--text-primary) !important;
    }}
    /* Franja de color decorativa que Streamlit pinta justo debajo del header. */
    div[data-testid="stDecoration"] {{
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    /* El contenedor raiz tambien puede mostrar blanco durante la carga inicial. */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg);
    }}

    h1, h2, h3, h4, h5, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}

    p, span, label, .stMarkdown, .stCaption, div[data-testid="stMarkdownContainer"] {{
        color: var(--text-primary);
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--text-secondary) !important;
    }}

    /* --- Tarjetas "glass" para todo container(border=True) --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s ease, transform 0.15s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: var(--accent);
    }}

    /* --- Botones --- */
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: #0B0F14;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        transition: transform 0.12s ease, box-shadow 0.2s ease;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(139,92,246,0.35);
    }}

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--text-secondary);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }}

    /* --- Metric --- */
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--accent);
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* --- Alerts (info/success/warning/error) con tinte glass --- */
    div[data-testid="stAlert"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        backdrop-filter: blur(8px);
    }}

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.02);
        border-right: 1px solid var(--border);
    }}

    /* --- Expander --- */
    details {{
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    /* --- Stepper de fases (custom, ver render_stepper) --- */
    .adm-stepper {{
        display: flex;
        gap: 0.4rem;
        margin: 0.5rem 0 1.2rem 0;
        flex-wrap: wrap;
    }}
    .adm-step {{
        flex: 1;
        min-width: 100px;
        padding: 0.5rem 0.7rem;
        border-radius: 10px;
        background: var(--surface);
        border: 1px solid var(--border);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        color: var(--text-secondary);
        text-align: center;
        transition: all 0.2s ease;
    }}
    .adm-step.active {{
        border-color: var(--accent);
        color: var(--text-primary);
        background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(34,211,238,0.10));
        box-shadow: 0 0 0 1px var(--accent) inset;
    }}
    .adm-step.done {{
        color: var(--accent-2);
        border-color: rgba(255,255,255,0.14);
    }}

    /* --- Badge de dominio --- */
    .adm-domain-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(34,211,238,0.12));
        border: 1px solid var(--accent);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        color: var(--text-primary);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return cfg


def render_stepper(phases: list[str], current_index: int) -> None:
    """Renderiza el stepper horizontal de fases con la fase activa resaltada."""
    items = []
    for i, label in enumerate(phases):
        cls = "active" if i == current_index else ("done" if i < current_index else "")
        items.append(f'<div class="adm-step {cls}">{i+1}. {label}</div>')
    st.markdown(f'<div class="adm-stepper">{"".join(items)}</div>',
               unsafe_allow_html=True)


def render_domain_badge(cfg: dict) -> None:
    st.markdown(
        f'<div class="adm-domain-badge">{cfg["icon"]} {cfg["label"]} &middot; '
        f'{cfg["tagline"]}</div>',
        unsafe_allow_html=True,
    )
