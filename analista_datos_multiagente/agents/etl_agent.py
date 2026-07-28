"""Capa 1 - Agente ETL con narracion pedagogica.

Responsabilidades:
- Detectar tipos de columna (numerica, categorica, fecha, texto, booleana).
- Detectar y reportar nulos (%), duplicados, outliers (IQR) e inconsistencias.
- Explicar la estrategia de limpieza ANTES de aplicarla y luego aplicarla.
- Dejar el dataset limpio versionado en el estado.

Toda decision se explica en lenguaje simple (via LLM o fallback local).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import data_loading as dl
from utils import visualization as viz
from utils.domain_config import vocab_for
from utils.llm import narrate_stream

AGENT = "ETL"


def _iqr_outliers(series: pd.Series) -> int:
    s = series.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


def run_etl(state) -> dict[str, Any]:
    bus = state["bus"]
    df: pd.DataFrame = state["raw_df"].copy()
    domain = state.get("domain", "general")
    vocab = vocab_for(domain)

    bus.agent_start(AGENT, "Agente ETL - limpieza y diagnostico de datos", icon="🧹")

    # ---------------- Diagnostico ----------------
    overview = dl.basic_overview(df)
    profiles = dl.profile_dataframe(df)
    by_kind = dl.columns_by_kind(profiles)

    pct_missing = {p.name: p.pct_missing for p in profiles}
    outliers = {
        p.name: _iqr_outliers(df[p.name])
        for p in profiles
        if p.dtype_kind == "numeric"
    }
    total_outliers = int(sum(outliers.values()))

    # Grafico de nulos y de tipos de columna.
    bus.chart(AGENT, "Valores faltantes", viz.missing_values_bar(pct_missing),
              caption="Porcentaje de celdas vacias en cada columna.")
    type_counts = {k: len(v) for k, v in by_kind.items() if v}
    bus.chart(AGENT, "Tipos de columna", viz.column_types_pie(type_counts),
              caption="Como interpreta el sistema cada columna.")

    high_missing = {c: v for c, v in pct_missing.items() if v > 0}
    resumen_diag = (
        f"El dataset tiene {overview['n_rows']} registros de {vocab['row_word']} y "
        f"{overview['n_cols']} columnas. Detecte {overview['n_duplicates']} "
        f"{vocab['duplicate_phrase']}, {overview['total_missing']} {vocab['null_phrase']} "
        f"en total y {total_outliers} {vocab['outlier_phrase']} en columnas numericas."
    )

    diag_prompt = (
        "Explica a una persona sin experiencia el estado inicial de su dataset. "
        f"Datos: {overview['n_rows']} filas, {overview['n_cols']} columnas, "
        f"{overview['n_duplicates']} duplicados, columnas con nulos: "
        f"{high_missing or 'ninguna'}, outliers detectados por columna: "
        f"{outliers or 'ninguno'}. Explica que significan los nulos, los duplicados "
        "y los outliers, y por que hay que tratarlos antes de entrenar modelos."
    )
    bus.narrate(AGENT, narrate_stream(diag_prompt, resumen_diag, domain=domain))

    # ---------------- Estrategia (explicada antes de aplicar) ----------------
    strategy_actions: list[str] = []
    for p in profiles:
        if p.pct_missing <= 0:
            continue
        if p.dtype_kind == "numeric":
            skew = abs(df[p.name].skew()) if df[p.name].dropna().size > 2 else 0
            method = "mediana" if skew > 1 else "media"
            strategy_actions.append(
                f"'{p.name}' ({p.pct_missing}% nulos): imputar con la {method} "
                f"({'distribucion sesgada' if method == 'mediana' else 'distribucion simetrica'})."
            )
        elif p.dtype_kind in ("categorical", "boolean"):
            strategy_actions.append(
                f"'{p.name}' ({p.pct_missing}% nulos): imputar con la moda (valor mas frecuente)."
            )
        elif p.dtype_kind == "datetime":
            strategy_actions.append(
                f"'{p.name}' ({p.pct_missing}% nulos): rellenar hacia adelante (orden temporal)."
            )
        else:
            strategy_actions.append(
                f"'{p.name}' ({p.pct_missing}% nulos): rellenar con la etiqueta 'desconocido'."
            )

    drop_cols = [p.name for p in profiles if p.pct_missing >= 60]
    if drop_cols:
        strategy_actions.append(
            f"Eliminar columnas con >=60% de nulos: {', '.join(drop_cols)} (aportan poca informacion)."
        )

    strategy_text = (
        "Antes de tocar nada, esta es mi estrategia de limpieza:\n- "
        + "\n- ".join(strategy_actions if strategy_actions else ["No hay nulos que imputar."])
        + f"\n- Eliminar {overview['n_duplicates']} filas duplicadas."
        + (f"\n- Recortar {total_outliers} outliers al rango valido (winsorizacion por IQR)."
           if total_outliers else "")
    )
    strat_prompt = (
        "Explica en lenguaje simple la estrategia de limpieza que vas a aplicar y "
        "por que eliges cada tecnica (media vs mediana vs moda, eliminar duplicados, "
        "tratar outliers con IQR). Se breve. Estrategia concreta:\n" + strategy_text
    )
    bus.narrate(AGENT, narrate_stream(strat_prompt, strategy_text, domain=domain))

    # ---------------- Aplicacion ----------------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dups_removed = n_before - len(df)

    if drop_cols:
        df = df.drop(columns=drop_cols)
        profiles = [p for p in profiles if p.name not in drop_cols]

    imputed: dict[str, str] = {}
    for p in profiles:
        col = p.name
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        if p.dtype_kind == "numeric":
            skew = abs(df[col].skew()) if df[col].dropna().size > 2 else 0
            value = df[col].median() if skew > 1 else df[col].mean()
            df[col] = df[col].fillna(value)
            imputed[col] = "mediana" if skew > 1 else "media"
        elif p.dtype_kind in ("categorical", "boolean"):
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
                imputed[col] = "moda"
        elif p.dtype_kind == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce").ffill().bfill()
            imputed[col] = "ffill"
        else:
            df[col] = df[col].fillna("desconocido")
            imputed[col] = "etiqueta 'desconocido'"

    # Winsorizacion de outliers (recorte al rango IQR), menos destructivo que borrar filas.
    winsorized: dict[str, int] = {}
    for col in by_kind.get("numeric", []):
        if col not in df.columns:
            continue
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        n_clip = int(mask.sum())
        if n_clip:
            df[col] = s.clip(lo, hi)
            winsorized[col] = n_clip

    clean_overview = dl.basic_overview(df)
    resumen_final = (
        f"Dataset limpio: {clean_overview['n_rows']} registros de {vocab['row_word']} x "
        f"{clean_overview['n_cols']} columnas. Elimine {n_dups_removed} "
        f"{vocab['duplicate_phrase']}, impute {len(imputed)} columnas y trate "
        f"{vocab['outlier_phrase']} en {len(winsorized)} columnas. Ya no quedan "
        f"{vocab['null_phrase']}: {clean_overview['total_missing'] == 0}."
    )
    bus.narrate(
        AGENT,
        narrate_stream(
            "Resume el resultado de la limpieza en 2-3 frases, en tono pedagogico y "
            "positivo, indicando que el dataset ya esta listo para analizar. Datos: "
            + resumen_final,
            resumen_final,
            domain=domain,
        ),
    )

    # Vista previa del dataset limpio.
    bus.table(AGENT, "Vista previa del dataset limpio", df.head(10))

    etl_report = {
        "overview_before": overview,
        "overview_after": clean_overview,
        "profiles": [p.__dict__ for p in profiles],
        "by_kind": by_kind,
        "pct_missing": pct_missing,
        "outliers": outliers,
        "duplicates_removed": n_dups_removed,
        "imputed": imputed,
        "winsorized": winsorized,
        "dropped_columns": drop_cols,
        "domain": domain,
    }

    bus.agent_end(AGENT, resumen_final)
    return {"clean_df": df, "etl_report": etl_report}
