"""Capa 10 - Simulador de escenarios estrategicos ("War Room").

Activo SOLO en el dominio Estrategia (ver app.py). Dos piezas, ambas apoyadas en
artefactos YA calculados por las capas 3/4 (nunca se reentrena nada aqui):

    1. Re-prediccion what-if: usa el modelo ya entrenado por el Agente
       Supervisado (`supervised_result["trained_model"]`) para predecir sobre un
       vector de entrada modificado por sliders. Requiere reconstruir el vector
       de features en el MISMO orden con el que se entreno (`build_features` +
       el `scaler` ya ajustado), partiendo de la fila promedio real del dataset
       y sustituyendo solo las columnas que el usuario mueve.

    2. Matriz BCG (participacion vs. crecimiento): usa los CLUSTERS ya
       calculados por el Agente No Supervisado (`unsupervised_result
       ["cluster_profile"]`) como "unidades de negocio". Los ejes de la matriz
       se eligen entre columnas numericas reales (no se inventa una metrica de
       "crecimiento" que el dataset no tenga); el LLM ejecutivo solo REFORMULA
       esos numeros en un cuadrante (nunca inventa una cifra nueva).
"""

from __future__ import annotations

from typing import Any

import pandas as pd




# ----------------------------------------------------------------------
# 1. Re-prediccion what-if sobre el modelo ya entrenado
# ----------------------------------------------------------------------
def build_baseline_row(df: pd.DataFrame, target: str | None) -> pd.Series:
    """Fila de referencia: la MEDIANA real de cada columna numerica y la MODA

    de cada columna categorica del dataset limpio. Es el punto de partida del
    what-if antes de mover ningun slider (ningun valor es inventado: todos
    salen de `clean_df`).
    """
    row: dict[str, Any] = {}
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            row[col] = float(s.median())
        elif pd.api.types.is_bool_dtype(s):
            row[col] = bool(s.mode(dropna=True).iloc[0]) if not s.mode().empty else False
        else:
            mode = s.mode(dropna=True)
            row[col] = mode.iloc[0] if not mode.empty else ""
    return pd.Series(row)


def predict_scenario(
    supervised_result: dict[str, Any],
    baseline_row: pd.Series,
    overrides: dict[str, float],
) -> dict[str, Any]:
    """Re-predice usando el modelo YA entrenado (`trained_model`), sin reentrenar.

    `overrides` es {columna: nuevo_valor} tomado de los sliders de la UI.
    Devuelve la prediccion (valor o clase) y, si es clasificacion binaria con
    predict_proba, la probabilidad de la clase positiva.
    """
    from agents.common import build_features

    model = supervised_result.get("trained_model")
    scaler = supervised_result.get("trained_scaler")
    feature_names = supervised_result.get("trained_feature_names")
    if model is None or scaler is None or not feature_names:
        return {"status": "unavailable",
                "reason": "No hay un modelo entrenado disponible en esta sesion."}

    scenario_row = baseline_row.copy()
    for col, val in overrides.items():
        if col in scenario_row.index:
            scenario_row[col] = val

    scenario_df = pd.DataFrame([scenario_row])
    feats = build_features(scenario_df)

    # Alineamos al MISMO orden/columnas con las que se entreno el modelo. Las
    # columnas de entrenamiento que no aparecen en este escenario (ej. una
    # categoria de one-hot ausente en la fila unica) se rellenan con 0.
    aligned = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for name in feats.feature_names:
        if name in aligned.columns:
            idx = feats.feature_names.index(name)
            aligned.loc[0, name] = feats.X[0, idx]

    X_scaled = scaler.transform(aligned.to_numpy(dtype=float))
    task = supervised_result.get("task")
    labels = supervised_result.get("labels")

    pred = model.predict(X_scaled)[0]
    out: dict[str, Any] = {"status": "ok", "raw_prediction": float(pred)
                            if task == "regresion" else pred}

    if task == "clasificacion" and labels:
        pred_label = labels[int(pred)] if int(pred) < len(labels) else str(pred)
        out["predicted_label"] = pred_label
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_scaled)[0]
                out["proba"] = {labels[i]: float(p) for i, p in enumerate(proba)
                                if i < len(labels)}
            except Exception:  # noqa: BLE001
                pass
    return out


# ----------------------------------------------------------------------
# 2. Matriz BCG sobre los clusters ya calculados
# ----------------------------------------------------------------------
def build_bcg_quadrants(
    unsupervised_result: dict[str, Any],
    x_axis: str,
    y_axis: str,
) -> list[dict[str, Any]]:
    """Ubica cada cluster (ya calculado) en la matriz BCG segun `x_axis`/`y_axis`.

    Ambos ejes deben ser columnas presentes en `cluster_profile` (medias reales
    por cluster). El cuadrante se asigna comparando cada cluster contra la MEDIA
    GLOBAL real de esas mismas columnas (`overall_mean`), sin ningun umbral
    inventado.
    """
    profile = unsupervised_result.get("cluster_profile", {})
    overall = unsupervised_result.get("overall_mean", {})
    sizes = unsupervised_result.get("cluster_sizes", {})
    if not profile or x_axis not in overall or y_axis not in overall:
        return []

    x_mid, y_mid = overall[x_axis], overall[y_axis]
    quadrants = []
    for cluster_id, values in profile.items():
        if x_axis not in values or y_axis not in values:
            continue
        x_val, y_val = values[x_axis], values[y_axis]
        high_x, high_y = x_val >= x_mid, y_val >= y_mid
        if high_x and high_y:
            label = "Estrella"
        elif high_x and not high_y:
            label = "Vaca lechera"
        elif not high_x and high_y:
            label = "Interrogante"
        else:
            label = "Perro"
        quadrants.append({
            "cluster": cluster_id,
            "label": label,
            "x": x_val,
            "y": y_val,
            "size": sizes.get(cluster_id, sizes.get(str(cluster_id), 0)),
        })
    return quadrants


def summarize_bcg_with_llm(
    quadrants: list[dict[str, Any]],
    x_axis: str,
    y_axis: str,
    domain: str,
) -> str:
    """El LLM ejecutivo REFORMULA los cuadrantes ya calculados; nunca inventa

    clusters ni cifras nuevas. Fallback determinista si no hay LLM.
    """
    fallback_parts = []
    for q in sorted(quadrants, key=lambda d: d["size"], reverse=True):
        fallback_parts.append(
            f"Cluster {q['cluster']} ({q['size']} registros) es '{q['label']}': "
            f"{x_axis}={q['x']:.2f}, {y_axis}={q['y']:.2f}."
        )
    fallback = " ".join(fallback_parts) or "No hay clusters suficientes para la matriz."

    prompt = (
        f"Resume en 3-4 frases, tono ejecutivo, esta matriz BCG calculada sobre "
        f"segmentos reales de datos. Eje X='{x_axis}', eje Y='{y_axis}'. "
        f"Datos (unica fuente permitida): {quadrants}. No inventes clusters ni "
        f"cifras que no esten en estos datos."
    )
    from .llm import narrate_text

    return narrate_text(prompt, fallback_text=fallback,
                        domain=(domain if domain != "general" else None))


def default_bcg_axes(overall_mean: dict[str, float], target: str | None) -> tuple[str, str]:
    """Elige 2 columnas numericas reales como ejes por defecto (nunca inventadas)."""
    cols = [c for c in overall_mean.keys() if c != target]
    if len(cols) >= 2:
        return cols[0], cols[1]
    if len(cols) == 1:
        return cols[0], cols[0]
    return "", ""
