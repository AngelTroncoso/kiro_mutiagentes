"""Capa 2 - Agente Router / Diagnosticador.

Analiza el dataset limpio y decide, con reglas transparentes, que enfoques de
Machine Learning tienen sentido y por que. Tambien propone una variable objetivo
por defecto. El resultado se usa en la UI para PRE-MARCAR un selector que el
usuario puede editar.

Este agente NO entrena modelos: solo diagnostica y explica.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.domain_config import frame_target, get_domain, is_domain_match
from utils.llm import narrate_stream

AGENT = "Router"


def _looks_categorical(series: pd.Series, n_rows: int) -> bool:
    if series.dtype == object or str(series.dtype) == "category" or series.dtype == bool:
        return True
    nun = series.nunique(dropna=True)
    # Numerica pero con pocos valores distintos -> probablemente clases codificadas.
    return nun <= max(10, int(0.05 * n_rows)) and nun <= 20


def suggest_target(clean_df: pd.DataFrame, by_kind: dict[str, list[str]]) -> str | None:
    """Heuristica para proponer la variable objetivo."""
    if clean_df.shape[1] == 0:
        return None
    # Nombres que suelen indicar objetivo.
    keywords = ("target", "label", "class", "objetivo", "resultado", "y",
                "price", "precio", "churn", "default", "outcome", "sales", "ventas")
    for col in clean_df.columns:
        if str(col).strip().lower() in keywords or any(
            k in str(col).lower() for k in keywords
        ):
            return str(col)
    # Si no, la ultima columna es una convencion habitual.
    return str(clean_df.columns[-1])


def run_router(state) -> dict[str, Any]:
    bus = state["bus"]
    df: pd.DataFrame = state["clean_df"]
    etl_report = state.get("etl_report", {})
    by_kind = etl_report.get("by_kind", {})
    n_rows, n_cols = df.shape
    domain = state.get("domain", "general")
    domain_cfg = get_domain(domain)

    bus.agent_start(AGENT, "Agente Router - diagnostico de enfoques de ML", icon="🧭")

    # El dominio SOLO enmarca lenguaje; nunca decide si el enfoque aplica.
    domain_match = is_domain_match(list(df.columns), domain)
    if domain != "general" and not domain_match:
        bus.warning(
            AGENT,
            f"Este dataset no parece calzar con columnas tipicas de "
            f"{domain_cfg['label']}. Sigo el diagnostico igual, pero interpreta "
            f"las sugerencias de dominio con cautela: podrian no corresponder.",
        )

    numeric = [c for c in by_kind.get("numeric", []) if c in df.columns]
    categorical = [c for c in by_kind.get("categorical", []) if c in df.columns]
    datetime_cols = [c for c in by_kind.get("datetime", []) if c in df.columns]

    target = state.get("target_column") or suggest_target(df, by_kind)

    # --- Determinar tarea supervisada ---
    supervised_ok = False
    supervised_task = None  # "clasificacion" | "regresion"
    supervised_reason = ""
    if target and target in df.columns and n_rows >= 20:
        tser = df[target]
        if _looks_categorical(tser, n_rows):
            supervised_ok = True
            supervised_task = "clasificacion"
            supervised_reason = (
                f"'{target}' tiene pocas categorias distintas ({tser.nunique()}), "
                "asi que se puede predecir a que grupo pertenece cada fila."
            )
        elif pd.api.types.is_numeric_dtype(tser):
            supervised_ok = True
            supervised_task = "regresion"
            supervised_reason = (
                f"'{target}' es un numero continuo, asi que se puede predecir su valor."
            )
    if not supervised_ok and not supervised_reason:
        supervised_reason = "No hay una variable objetivo clara con suficientes datos."

    # --- No supervisado (clustering) ---
    unsupervised_ok = len(numeric) >= 2 and n_rows >= 20
    unsupervised_reason = (
        f"Hay {len(numeric)} columnas numericas y {n_rows} filas, suficiente para "
        "buscar agrupaciones naturales (clusters)."
        if unsupervised_ok
        else "Se necesitan al menos 2 columnas numericas y 20 filas para agrupar."
    )

    # --- Refuerzo (aproximacion: bandit contextual sobre el objetivo) ---
    has_temporal = len(datetime_cols) >= 1
    reinforcement_ok = supervised_ok and n_rows >= 40
    reinforcement_reason = (
        "Se puede simular un problema de decision secuencial: el agente aprende que "
        "'accion' (categoria/rango del objetivo) conviene segun el contexto de cada fila."
        + (" Ademas hay una columna temporal que refuerza esta idea."
           if has_temporal else " (No hay estructura temporal real; sera una aproximacion.)")
        if reinforcement_ok
        else "Sin una variable objetivo utilizable no se puede definir la recompensa."
    )

    applicable: list[str] = []
    if supervised_ok:
        applicable.append("supervised")
    if unsupervised_ok:
        applicable.append("unsupervised")
    if reinforcement_ok:
        applicable.append("reinforcement")

    # Enmarca el NOMBRE de la columna objetivo ya detectada con el lenguaje del
    # dominio (ej. una columna de churn -> "riesgo de fuga de ingresos" en
    # Finanzas). Nunca cambia cual es el objetivo ni inventa una columna nueva.
    target_frame = frame_target(target, domain) if supervised_ok else None

    diagnosis = {
        "target_column": target,
        "target_frame": target_frame,
        "domain": domain,
        "domain_match": domain_match,
        "supervised": {"applicable": supervised_ok, "task": supervised_task,
                       "reason": supervised_reason},
        "unsupervised": {"applicable": unsupervised_ok, "reason": unsupervised_reason},
        "reinforcement": {"applicable": reinforcement_ok, "reason": reinforcement_reason,
                          "has_temporal": has_temporal},
        "applicable": applicable,
        "columns": {"numeric": numeric, "categorical": categorical,
                    "datetime": datetime_cols},
    }

    # --- Narracion pedagogica del diagnostico ---
    target_desc = (f"'{target}' (se enmarca como {target_frame})"
                   if target_frame else f"'{target}'")
    fallback = (
        f"Revise el dataset limpio. La variable objetivo mas probable es {target_desc}. "
        + (f"Como {supervised_reason.lower()} " if supervised_ok else "")
        + ("Recomiendo aprendizaje SUPERVISADO"
           + (f" ({supervised_task}). " if supervised_task else ". ")
           if supervised_ok else "No veo un objetivo claro para aprendizaje supervisado. ")
        + ("Tambien tiene sentido el NO SUPERVISADO para descubrir grupos. "
           if unsupervised_ok else "")
        + ("Y podemos probar una aproximacion de REFUERZO como problema de decision. "
           if reinforcement_ok else "")
        + "Deje pre-seleccionados los analisis recomendados, pero puedes cambiarlos."
    )
    prompt = (
        "Explica en lenguaje simple que tipos de Machine Learning tienen sentido para "
        "este dataset y por que, para alguien que recien empieza. Diferencia supervisado "
        "(predecir con ejemplos etiquetados), no supervisado (descubrir grupos sin "
        "etiquetas) y refuerzo (aprender por prueba y recompensa). "
        f"Diagnostico: objetivo probable='{target}' (enmarcado como: {target_frame}); "
        f"supervisado={supervised_ok} ({supervised_task}, {supervised_reason}); "
        f"no_supervisado={unsupervised_ok} ({unsupervised_reason}); "
        f"refuerzo={reinforcement_ok} ({reinforcement_reason})."
    )
    bus.narrate(AGENT, narrate_stream(prompt, fallback, domain=domain))

    bus.result(AGENT, "diagnosis", {
        "target": target,
        "target_frame": target_frame,
        "applicable": applicable,
        "supervised_task": supervised_task,
    })
    bus.agent_end(AGENT, f"Enfoques recomendados: {', '.join(applicable) or 'ninguno'}.")

    return {"diagnosis": diagnosis, "target_column": target}
