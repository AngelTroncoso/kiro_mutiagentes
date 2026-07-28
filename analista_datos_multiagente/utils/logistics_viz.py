"""Capa 9 - Rutas e inventario animado (activo solo en dominio Logistica).

Dos visualizaciones independientes, cada una con su propia degradacion segura:

    1. Mapa de rutas (pydeck ArcLayer): requiere columnas de coordenadas reales
       (lat/lon) o un mapeo manual que el USUARIO ingrese explicitamente para
       cada valor categorico (ej. ciudad/ruta -> lat/lon). Nunca se inventan
       coordenadas por defecto.
    2. Animacion de inventario (Plotly frames): requiere una columna de stock y,
       opcionalmente, una de tiempo/orden. El punto de reorden (ROP) se deriva
       con un metodo estadistico simple y trazable (percentil bajo de la propia
       distribucion real de stock), nunca una formula de demanda inventada.

Si ninguna de las dos aplica, la UI debe mostrar un aviso claro (ver app.py) en
vez de forzar una animacion vacia.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .visualization import _accent, _active_template  # reuso del tema oscuro

# --- Deteccion de columnas (solo por nombre; nunca infiere valores) ---------
_LAT_KEYWORDS = ("lat", "latitud", "latitude")
_LON_KEYWORDS = ("lon", "lng", "longitud", "longitude")
_CATEGORICAL_LOCATION_KEYWORDS = ("ciudad", "ruta", "region", "región", "destino",
                                 "origen", "zona", "sede", "almacen", "almacén")
_STOCK_KEYWORDS = ("stock", "inventario", "existencia", "unidades_disponibles")
_SLA_KEYWORDS = ("retraso", "delay", "incumpl", "sla")


def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Busca columnas numericas cuyo NOMBRE sugiera latitud/longitud."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    lat_col = next((c for c in numeric_cols if any(k in str(c).lower()
                    for k in _LAT_KEYWORDS)), None)
    lon_col = next((c for c in numeric_cols if any(k in str(c).lower()
                    for k in _LON_KEYWORDS)), None)
    return lat_col, lon_col


def detect_categorical_location_column(df: pd.DataFrame) -> str | None:
    """Columna categorica que probablemente representa una ubicacion (ciudad/ruta)."""
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in _CATEGORICAL_LOCATION_KEYWORDS):
            if not pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def detect_stock_column(df: pd.DataFrame) -> str | None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return next((c for c in numeric_cols if any(k in str(c).lower()
                for k in _STOCK_KEYWORDS)), None)


def detect_delay_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(k in str(col).lower() for k in _SLA_KEYWORDS):
            return col
    return None


# --- 1. Mapa de rutas (pydeck) ----------------------------------------------
def build_routes_map(
    df: pd.DataFrame,
    location_col: str,
    manual_coords: dict[str, tuple[float, float]],
    delay_col: str | None = None,
    max_routes: int = 200,
):
    """Construye un pydeck.Deck con ArcLayer entre un origen fijo y cada destino.

    `manual_coords` es un mapeo {valor_categorico: (lat, lon)} que el USUARIO
    ingreso explicitamente en la UI. Ninguna coordenada se inventa aqui: si un
    valor de `location_col` no esta en `manual_coords`, esa fila se descarta.
    """
    import pydeck as pdk

    sample = df[[location_col] + ([delay_col] if delay_col else [])].dropna(
        subset=[location_col]).head(max_routes).copy()
    sample["_lat"] = sample[location_col].map(lambda v: manual_coords.get(str(v), (None, None))[0])
    sample["_lon"] = sample[location_col].map(lambda v: manual_coords.get(str(v), (None, None))[1])
    sample = sample.dropna(subset=["_lat", "_lon"])
    if sample.empty:
        return None, 0

    # Origen: centroide de los destinos mapeados (no un supuesto arbitrario).
    origin_lat = float(sample["_lat"].mean())
    origin_lon = float(sample["_lon"].mean())

    if delay_col and delay_col in sample.columns:
        sample["_delayed"] = pd.to_numeric(sample[delay_col], errors="coerce").fillna(0) > 0
    else:
        sample["_delayed"] = False

    records = [
        {
            "from_lon": origin_lon, "from_lat": origin_lat,
            "to_lon": float(row["_lon"]), "to_lat": float(row["_lat"]),
            "destino": str(row[location_col]),
            "color": [255, 107, 107] if row["_delayed"] else [34, 211, 238],
        }
        for _, row in sample.iterrows()
    ]

    layer = pdk.Layer(
        "ArcLayer",
        data=records,
        get_source_position="[from_lon, from_lat]",
        get_target_position="[to_lon, to_lat]",
        get_source_color=[139, 92, 246],
        get_target_color="color",
        get_width=3,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=origin_lat, longitude=origin_lon, zoom=4, pitch=30)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                    map_style=None, tooltip={"text": "Destino: {destino}"})
    return deck, len(records)


# --- 2. Animacion de inventario (Plotly frames) -----------------------------
def compute_reorder_point(stock: pd.Series, percentile: float = 20.0) -> dict[str, float]:
    """Punto de reorden (ROP) como percentil bajo de la distribucion REAL de stock.

    Metodo deliberadamente simple y transparente (no una formula de demanda que
    requeriria datos de consumo diario que este dataset no tiene). Se declara la
    metodologia explicitamente en la UI para que no se confunda con un ROP
    calculado por demanda x lead time.
    """
    s = pd.to_numeric(stock, errors="coerce").dropna()
    if s.empty:
        return {"rop": 0.0, "percentile": percentile, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "rop": float(np.percentile(s, percentile)),
        "percentile": percentile,
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def inventory_animation(stock: pd.Series, rop: float, max_frames: int = 120) -> go.Figure:
    """Grafico animado (frames) de la evolucion del stock vs el punto de reorden.

    El eje X es el ORDEN de las filas en el dataset (unico eje disponible sin
    columna de fecha real); se declara asi en el caption de la UI.
    """
    s = pd.to_numeric(stock, errors="coerce").dropna().reset_index(drop=True)
    if len(s) > max_frames:
        # Muestreo uniforme para no generar cientos de frames.
        idx = np.linspace(0, len(s) - 1, max_frames).astype(int)
        s = s.iloc[idx].reset_index(drop=True)

    x = list(range(len(s)))
    below_rop = s < rop

    frames = []
    for i in range(1, len(s) + 1):
        frames.append(go.Frame(
            data=[go.Scatter(x=x[:i], y=s[:i].tolist(), mode="lines+markers",
                             line=dict(color=_accent(0)),
                             marker=dict(
                                 color=[_accent(3) if b else _accent(0)
                                       for b in below_rop[:i]]))],
            name=str(i),
        ))

    fig = go.Figure(
        data=[go.Scatter(x=x[:1], y=s[:1].tolist(), mode="lines+markers",
                         line=dict(color=_accent(0)))],
        frames=frames,
    )
    fig.add_hline(y=rop, line_dash="dash", line_color=_accent(3),
                  annotation_text=f"Punto de reorden (P20 real) = {rop:,.0f}")

    fig.update_layout(
        template=_active_template(),
        title="Evolucion del inventario vs. punto de reorden",
        xaxis_title="Secuencia de registros en el dataset",
        yaxis_title="Stock disponible",
        height=440,
        margin=dict(l=40, r=20, t=60, b=40),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶ Reproducir", "method": "animate",
                 "args": [None, {"frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True}]},
                {"label": "⏸ Pausar", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate"}]},
            ],
        }],
    )
    # Frame final visible por defecto (para quien no le da play).
    if frames:
        fig.update_traces(x=x, y=s.tolist())
    return fig


def availability_report(df: pd.DataFrame) -> dict[str, Any]:
    """Resume que visualizaciones son posibles con las columnas REALES del df."""
    lat_col, lon_col = detect_lat_lon_columns(df)
    cat_loc_col = detect_categorical_location_column(df)
    stock_col = detect_stock_column(df)
    delay_col = detect_delay_column(df)
    return {
        "has_real_coords": bool(lat_col and lon_col),
        "lat_col": lat_col,
        "lon_col": lon_col,
        "categorical_location_col": cat_loc_col,
        "stock_col": stock_col,
        "delay_col": delay_col,
        "map_possible": bool((lat_col and lon_col) or cat_loc_col),
        "inventory_possible": bool(stock_col),
    }
