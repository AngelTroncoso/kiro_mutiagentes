"""Estado compartido que fluye por el grafo de agentes (Capa 0).

El estado es un TypedDict porque LangGraph lo usa como esquema del grafo. Los
DataFrames y objetos no serializables se guardan directamente (no persistimos el
grafo en disco, la ejecucion vive en memoria durante una corrida).

Flujo del dataset:
    raw_df  --(ETL)-->  clean_df  --(agentes ML)-->  resultados por enfoque
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from utils.streaming import EventBus


class PipelineState(TypedDict, total=False):
    # --- Infraestructura ---
    bus: EventBus                       # canal de streaming hacia la UI
    config: dict[str, Any]              # opciones elegidas por el usuario

    # --- Capa 0: dominio de negocio (Finanzas/Logistica/Estrategia/General) ---
    # Es una capa de PRESENTACION: reconfigura vocabulario y sugerencias, nunca
    # la matematica (metricas, modelos, scores son identicos entre dominios).
    domain: str

    # --- Datos ---
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame

    # --- Reportes por capa ---
    etl_report: dict[str, Any]          # diagnostico + acciones de limpieza
    diagnosis: dict[str, Any]           # salida del router (que ML aplica y por que)

    # --- Seleccion del usuario (capa 2) ---
    selected_analyses: list[str]        # subconjunto de {"supervised","unsupervised","reinforcement"}
    target_column: str | None           # variable objetivo (si aplica)
    business_goal: str | None            # objetivo de negocio para el recomendador

    # --- Resultados de la capa 3 ---
    supervised_result: dict[str, Any]
    unsupervised_result: dict[str, Any]
    reinforcement_result: dict[str, Any]

    # --- Recomendacion final (capa 4) ---
    recommendation: dict[str, Any]

    # --- Reporte ejecutivo gerencial (capa 6) ---
    executive_report: dict[str, Any]


def new_state(
    raw_df: pd.DataFrame,
    bus: EventBus,
    config: dict[str, Any] | None = None,
    domain: str = "general",
) -> PipelineState:
    return PipelineState(
        raw_df=raw_df,
        bus=bus,
        config=config or {},
        domain=domain,
        selected_analyses=[],
        target_column=None,
        business_goal=None,
    )
