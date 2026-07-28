"""Fabricas de graficos Plotly interactivos usados por los agentes.

El estilo visual (colores, fondo, tipografia) NO se fija aqui con valores sueltos:
se apoya en la plantilla oscura registrada por `utils.theme.apply_theme()`, para
que toda la app comparta una identidad visual consistente que ademas se adapta al
acento de color del dominio de negocio activo (Finanzas/Logistica/Estrategia).
Si el tema aun no fue aplicado (ej. en pruebas headless sin Streamlit), cae a una
plantilla clara neutra para no romper nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Paleta de respaldo (solo si el tema oscuro custom no fue registrado todavia).
PALETTE = px.colors.qualitative.Set2

_DARK_TEMPLATE_NAME = "analista_dark"
_ACCENT_FALLBACK = "#3b7dd8"
_ACCENT2_FALLBACK = "#ef8354"


def _active_template() -> str:
    """Usa la plantilla oscura custom si ya fue registrada; si no, un fallback claro."""
    return _DARK_TEMPLATE_NAME if _DARK_TEMPLATE_NAME in pio.templates else "plotly_white"


def _active_colorway() -> list[str]:
    if _DARK_TEMPLATE_NAME in pio.templates:
        tpl = pio.templates[_DARK_TEMPLATE_NAME]
        if tpl.layout.colorway:
            return list(tpl.layout.colorway)
    return list(PALETTE)


def _accent(idx: int = 0) -> str:
    colors = _active_colorway()
    return colors[idx % len(colors)] if colors else _ACCENT_FALLBACK


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template=_active_template(),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        font=dict(size=13),
    )
    return fig


def missing_values_bar(pct_missing: dict[str, float]) -> go.Figure:
    items = sorted(pct_missing.items(), key=lambda kv: kv[1], reverse=True)
    cols = [k for k, v in items if v > 0]
    vals = [pct_missing[c] for c in cols]
    if not cols:
        cols, vals = ["(sin nulos)"], [0]
    fig = go.Figure(go.Bar(x=vals, y=cols, orientation="h", marker_color=_accent(2)))
    fig.update_layout(xaxis_title="% de valores nulos", yaxis_title="")
    return _base_layout(fig, "Valores faltantes por columna")


def column_types_pie(counts: dict[str, int]) -> go.Figure:
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45,
                           marker=dict(colors=_active_colorway())))
    return _base_layout(fig, "Composicion de tipos de columna")


def feature_importance_bar(names: list[str], importances: list[float],
                           title: str = "Importancia de variables") -> go.Figure:
    order = np.argsort(importances)
    names = [names[i] for i in order]
    importances = [importances[i] for i in order]
    fig = go.Figure(go.Bar(x=importances, y=names, orientation="h",
                           marker_color=_accent(0)))
    fig.update_layout(xaxis_title="Importancia relativa", yaxis_title="")
    return _base_layout(fig, title)


def confusion_matrix_heatmap(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"pred {l}" for l in labels],
            y=[f"real {l}" for l in labels],
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    return _base_layout(fig, "Matriz de confusion")


def regression_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color=_accent(0), opacity=0.6),
                             name="Predicciones"))
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=_accent(1), dash="dash"),
                             name="Prediccion perfecta"))
    fig.update_layout(xaxis_title="Valor real", yaxis_title="Valor predicho")
    return _base_layout(fig, "Real vs. predicho")


def elbow_plot(ks: list[int], inertias: list[float], silhouettes: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                             name="Inercia", line=dict(color=_accent(0)),
                             yaxis="y1"))
    fig.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers",
                             name="Silhouette", line=dict(color=_accent(1)),
                             yaxis="y2"))
    fig.update_layout(
        xaxis_title="Numero de clusters (k)",
        yaxis=dict(title="Inercia", side="left"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right",
                    range=[0, 1]),
    )
    return _base_layout(fig, "Seleccion de k: metodo del codo + silhouette")


def cluster_scatter(coords: np.ndarray, labels: np.ndarray,
                    dims: int = 2) -> go.Figure:
    df = pd.DataFrame(coords[:, :max(dims, 2)],
                      columns=[f"PC{i+1}" for i in range(max(dims, 2))])
    df["Cluster"] = [f"Cluster {l}" for l in labels]
    if dims >= 3 and coords.shape[1] >= 3:
        df["PC3"] = coords[:, 2]
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="Cluster",
                            color_discrete_sequence=_active_colorway(), opacity=0.75)
    else:
        fig = px.scatter(df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=_active_colorway(), opacity=0.8)
    return _base_layout(fig, "Clusters proyectados con PCA")


def reward_curve(rewards: list[float], baseline: float | None = None) -> go.Figure:
    steps = list(range(1, len(rewards) + 1))
    cum = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=cum, mode="lines",
                             name="Recompensa media acumulada",
                             line=dict(color=_accent(0))))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=[steps[0], steps[-1]], y=[baseline, baseline],
                                 mode="lines", name="Politica aleatoria (baseline)",
                                 line=dict(color=_accent(1), dash="dash")))
    fig.update_layout(xaxis_title="Iteracion", yaxis_title="Recompensa media")
    return _base_layout(fig, "Aprendizaje del agente por refuerzo")


def recommendation_radar(scores: dict[str, float]) -> go.Figure:
    cats = list(scores.keys())
    vals = list(scores.values())
    cats_closed = cats + cats[:1]
    vals_closed = vals + vals[:1]
    fig = go.Figure(go.Scatterpolar(r=vals_closed, theta=cats_closed, fill="toself",
                                     line=dict(color=_accent(0))))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=False)
    return _base_layout(fig, "Puntaje normalizado por enfoque")
