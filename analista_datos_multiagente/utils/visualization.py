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


def cashflow_fan_chart(paths: np.ndarray, initial_capital: float) -> go.Figure:
    """Banda de confianza (fan chart) del flujo de caja acumulado simulado.

    `paths` es la matriz (n_iteraciones, horizonte+1) que devuelve el motor
    Monte Carlo de `utils.finance_sim`. Se dibujan los percentiles P10/P50/P90
    por periodo, mas una muestra de trayectorias individuales para dar
    sensacion de "animacion" de la incertidumbre sin depender de video.
    """
    periods = list(range(paths.shape[1]))
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    fig = go.Figure()

    # Muestra tenue de trayectorias individuales (sensacion de "banda viva").
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(paths.shape[0], size=min(40, paths.shape[0]), replace=False)
    for i in sample_idx:
        fig.add_trace(go.Scatter(x=periods, y=paths[i], mode="lines",
                                 line=dict(color=_accent(0), width=1),
                                 opacity=0.06, showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=periods + periods[::-1], y=list(p90) + list(p10[::-1]),
                             fill="toself", fillcolor="rgba(139,92,246,0.15)",
                             line=dict(color="rgba(0,0,0,0)"), name="Rango P10-P90",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=periods, y=p50, mode="lines+markers",
                             name="Mediana (P50)", line=dict(color=_accent(0), width=3)))
    fig.add_trace(go.Scatter(x=[periods[0], periods[-1]],
                             y=[initial_capital, initial_capital],
                             mode="lines", name="Capital inicial",
                             line=dict(color=_accent(2), dash="dot")))
    fig.add_trace(go.Scatter(x=[periods[0], periods[-1]], y=[0, 0], mode="lines",
                             name="Punto de quiebre (caja = 0)",
                             line=dict(color=_accent(3), dash="dash")))

    fig.update_layout(xaxis_title="Periodo", yaxis_title="Flujo de caja acumulado")
    return _base_layout(fig, "Proyeccion de flujo de caja (Monte Carlo)")


def npv_distribution_hist(npv_samples: np.ndarray) -> go.Figure:
    """Histograma de los VAN simulados, con las lineas P10/P50/P90 marcadas."""
    p10, p50, p90 = (float(x) for x in np.percentile(npv_samples, [10, 50, 90]))
    fig = go.Figure(go.Histogram(x=npv_samples, marker_color=_accent(0), nbinsx=60,
                                 opacity=0.85))
    for value, label, color_idx in ((p10, "P10", 1), (p50, "P50 (mediana)", 0),
                                     (p90, "P90", 2)):
        fig.add_vline(x=value, line_color=_accent(color_idx), line_dash="dash",
                      annotation_text=f"{label}: {value:,.0f}")
    fig.add_vline(x=0, line_color=_accent(3), line_width=2,
                  annotation_text="VAN = 0")
    fig.update_layout(xaxis_title="VAN (Valor Actual Neto) simulado",
                      yaxis_title="Frecuencia")
    return _base_layout(fig, "Distribucion del VAN sobre las simulaciones")


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
