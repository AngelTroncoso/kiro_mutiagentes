"""Capa 3 - Agente Supervisado (clasificacion o regresion).

Segun el tipo de variable objetivo:
- Categorica  -> clasificacion (RandomForest, LogisticRegression, GradientBoosting)
- Continua    -> regresion (RandomForest, LinearRegression, GradientBoosting)

Compara varios modelos, elige el mejor por su metrica principal y emite metricas,
grafico de importancia de variables y una explicacion pedagogica.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import visualization as viz
from utils.llm import narrate_stream

from .common import FeatureMatrix, build_features, encode_target

AGENT = "Supervisado"


def _permutation_importance(model, X, y, feature_names, scorer, n_repeats=5):
    """Importancia por permutacion, valida para cualquier modelo."""
    rng = np.random.default_rng(42)
    base = scorer(y, model.predict(X))
    importances = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - scorer(y, model.predict(Xp)))
        importances.append(max(np.mean(drops), 0.0))
    total = sum(importances) or 1.0
    return [i / total for i in importances]


def run_supervised(state) -> dict[str, Any]:
    bus = state["bus"]
    df: pd.DataFrame = state["clean_df"]
    diagnosis = state.get("diagnosis", {})
    target = state.get("target_column") or diagnosis.get("target_column")

    bus.agent_start(AGENT, "Agente Supervisado - aprender a predecir", icon="🎯")

    if not target or target not in df.columns:
        bus.warning(AGENT, "No hay variable objetivo utilizable; se omite el analisis supervisado.")
        bus.agent_end(AGENT, "Omitido: sin objetivo.")
        return {"supervised_result": {"status": "skipped", "reason": "sin objetivo"}}

    y_raw = df[target]
    y, labels = encode_target(y_raw)
    is_classification = labels is not None

    feats: FeatureMatrix = build_features(df, exclude=[target])
    X = StandardScaler().fit_transform(feats.X)

    if X.shape[1] == 0 or len(np.unique(y)) < 2 and is_classification:
        bus.warning(AGENT, "No hay suficientes variables o clases para entrenar.")
        bus.agent_end(AGENT, "Omitido: datos insuficientes.")
        return {"supervised_result": {"status": "skipped", "reason": "datos insuficientes"}}

    stratify = y if is_classification and np.min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # --- Definir modelos candidatos ---
    if is_classification:
        candidates = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Regresion Logistica": LogisticRegression(max_iter=1000),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        primary_metric = "F1 (macro)"
    else:
        candidates = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Regresion Lineal": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        primary_metric = "R2"

    # Narracion previa: que vamos a hacer.
    intro = (
        f"Voy a entrenar modelos para predecir '{target}'. "
        + (f"Como es una variable de categorias, es un problema de CLASIFICACION "
           f"(elegir a que grupo pertenece cada fila). " if is_classification else
           f"Como es un numero continuo, es un problema de REGRESION (estimar su valor). ")
        + f"Comparare {len(candidates)} modelos y me quedare con el mejor segun {primary_metric}."
    )
    bus.narrate(AGENT, narrate_stream(
        "Explica brevemente, para un principiante, que es el aprendizaje supervisado y "
        f"que vas a hacer ahora. Contexto: {intro}", intro))

    # --- Entrenar y evaluar ---
    results: dict[str, dict[str, float]] = {}
    best_name, best_model, best_score = None, None, -np.inf
    for name, model in candidates.items():
        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            if is_classification:
                metrics = {
                    "accuracy": float(accuracy_score(y_te, pred)),
                    "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
                }
                if len(labels) == 2 and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_te)[:, 1]
                        metrics["auc"] = float(roc_auc_score(y_te, proba))
                    except Exception:  # noqa: BLE001
                        pass
                score = metrics["f1_macro"]
            else:
                metrics = {
                    "r2": float(r2_score(y_te, pred)),
                    "mae": float(mean_absolute_error(y_te, pred)),
                    "rmse": float(np.sqrt(np.mean((y_te - pred) ** 2))),
                }
                score = metrics["r2"]
            results[name] = metrics
            if score > best_score:
                best_score, best_name, best_model = score, name, model
        except Exception as exc:  # noqa: BLE001
            bus.warning(AGENT, f"El modelo '{name}' fallo: {exc}")

    if best_model is None:
        bus.agent_end(AGENT, "No se pudo entrenar ningun modelo.")
        return {"supervised_result": {"status": "failed"}}

    # Tabla comparativa.
    comp_df = pd.DataFrame(results).T.round(4)
    bus.table(AGENT, "Comparacion de modelos", comp_df.reset_index().rename(
        columns={"index": "Modelo"}))

    # --- Importancia de variables ---
    pred_best = best_model.predict(X_te)
    if hasattr(best_model, "feature_importances_"):
        importances = list(best_model.feature_importances_)
    elif hasattr(best_model, "coef_"):
        coef = np.ravel(best_model.coef_)
        coef = coef[: len(feats.feature_names)] if coef.size >= len(feats.feature_names) else coef
        importances = list(np.abs(coef) / (np.abs(coef).sum() or 1.0))
    else:
        scorer = (lambda a, b: f1_score(a, b, average="macro", zero_division=0)) \
            if is_classification else r2_score
        importances = _permutation_importance(best_model, X_te, y_te, feats.feature_names, scorer)

    n = min(len(feats.feature_names), len(importances))
    top_names, top_imp = feats.feature_names[:n], importances[:n]
    order = np.argsort(top_imp)[-12:]
    bus.chart(AGENT, "Importancia de variables",
              viz.feature_importance_bar([top_names[i] for i in order],
                                         [top_imp[i] for i in order]),
              caption="Que columnas pesan mas en la prediccion del mejor modelo.")

    # Grafico especifico segun tarea.
    if is_classification:
        cm = confusion_matrix(y_te, pred_best)
        bus.chart(AGENT, "Matriz de confusion",
                  viz.confusion_matrix_heatmap(cm, labels),
                  caption="Aciertos (diagonal) vs. confusiones entre clases.")
    else:
        bus.chart(AGENT, "Real vs. predicho",
                  viz.regression_scatter(np.asarray(y_te), np.asarray(pred_best)),
                  caption="Cuanto mas cerca de la linea, mejor la prediccion.")

    best_metrics = results[best_name]
    top_feature = top_names[int(np.argmax(top_imp))] if top_imp else "(n/d)"

    # --- Explicacion pedagogica del resultado ---
    if is_classification:
        metric_txt = (f"acierta el {best_metrics['accuracy']*100:.1f}% de los casos "
                      f"(F1 macro {best_metrics['f1_macro']:.2f})")
        score_norm = best_metrics["f1_macro"]
    else:
        metric_txt = (f"explica el {best_metrics['r2']*100:.1f}% de la variacion "
                      f"(R2={best_metrics['r2']:.2f}), con error medio {best_metrics['mae']:.2f}")
        score_norm = max(0.0, best_metrics["r2"])

    fallback = (
        f"El mejor modelo fue '{best_name}': {metric_txt}. La variable mas influyente "
        f"para predecir '{target}' fue '{top_feature}'. "
        + ("Un F1 alto significa que el modelo equilibra bien aciertos entre todas las clases."
           if is_classification else
           "Un R2 cercano a 1 significa que el modelo captura bien el patron de los datos.")
    )
    bus.narrate(AGENT, narrate_stream(
        "Explica el resultado del modelo supervisado a un principiante: que significa la "
        f"metrica principal y por que importa. Resultado: {fallback}", fallback))

    result = {
        "status": "ok",
        "task": "clasificacion" if is_classification else "regresion",
        "target": target,
        "best_model": best_name,
        "primary_metric": primary_metric,
        "metrics": best_metrics,
        "all_models": results,
        "top_feature": top_feature,
        "score_norm": float(np.clip(score_norm, 0, 1)),
    }
    bus.result(AGENT, "supervised", result)
    bus.agent_end(AGENT, f"Mejor modelo: {best_name} ({primary_metric}={best_score:.3f}).")
    return {"supervised_result": result}
