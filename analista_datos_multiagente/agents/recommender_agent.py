"""Capa 4 - Agente Evaluador / Recomendador.

Recibe los resultados de los agentes ejecutados y recomienda que enfoque conviene
seguir usando, condicionado al OBJETIVO DE NEGOCIO que indique el usuario.

Problema clave: no se pueden comparar directamente metricas de distinta naturaleza
(accuracy vs. silhouette vs. recompensa). Solucion:
  1. Cada agente ya expone un `score_norm` en [0,1] (calidad interna del enfoque).
  2. El recomendador aplica PESOS segun el objetivo de negocio elegido, porque el
     "mejor" modelo depende de la pregunta que se quiere responder:
        - "predecir"  -> prioriza supervisado
        - "segmentar" -> prioriza no supervisado
        - "decidir"   -> prioriza refuerzo
        - "explorar"  -> pesos equilibrados
  3. El score final = score_norm * peso_objetivo. Gana el mayor, y se explica por que.
"""

from __future__ import annotations

from typing import Any

from utils import visualization as viz
from utils.domain_config import goal_labels_for
from utils.llm import narrate_stream

AGENT = "Recomendador"

# Pesos por objetivo de negocio. Filas = objetivo, columnas = enfoque.
# ESTOS PESOS SON INDEPENDIENTES DEL DOMINIO: el dominio solo cambia el TEXTO que
# ve el usuario (GOAL_LABELS -> domain_config.goal_labels_for). La matematica del
# recomendador (estos numeros) es identica sea cual sea el dominio elegido.
GOAL_WEIGHTS: dict[str, dict[str, float]] = {
    "predecir": {"supervised": 1.0, "unsupervised": 0.4, "reinforcement": 0.7},
    "segmentar": {"supervised": 0.4, "unsupervised": 1.0, "reinforcement": 0.3},
    "decidir": {"supervised": 0.7, "unsupervised": 0.3, "reinforcement": 1.0},
    "explorar": {"supervised": 0.7, "unsupervised": 0.7, "reinforcement": 0.6},
}

# Etiquetas neutras (sin dominio de negocio). Se usan como fallback y como fuente
# de las claves validas de `business_goal`. La UI, en cambio, muestra las
# etiquetas curadas por dominio via `goal_labels_for(domain)` (mismas claves).
GOAL_LABELS = {
    "predecir": "Predecir un valor o categoria concreta",
    "segmentar": "Descubrir grupos o segmentos",
    "decidir": "Optimizar una decision o accion",
    "explorar": "Explorar el dataset sin objetivo fijo",
}

APPROACH_LABELS = {
    "supervised": "Aprendizaje supervisado",
    "unsupervised": "Aprendizaje no supervisado",
    "reinforcement": "Aprendizaje por refuerzo (bandit)",
}


def run_recommender(state) -> dict[str, Any]:
    bus = state["bus"]
    domain = state.get("domain", "general")
    goal_labels = goal_labels_for(domain)  # solo texto; misma clave interna
    business_goal = (state.get("business_goal") or "explorar").lower()
    if business_goal not in GOAL_WEIGHTS:
        business_goal = "explorar"
    weights = GOAL_WEIGHTS[business_goal]  # <- matematica: SIEMPRE la misma

    bus.agent_start(AGENT, "Agente Recomendador - que enfoque conviene seguir", icon="🏆")

    # Recolectar score_norm de cada enfoque ejecutado con exito.
    raw_scores: dict[str, float] = {}
    details: dict[str, str] = {}

    sup = state.get("supervised_result") or {}
    if sup.get("status") == "ok":
        raw_scores["supervised"] = float(sup.get("score_norm", 0.0))
        details["supervised"] = (
            f"{sup.get('best_model')} ({sup.get('primary_metric')}="
            f"{sup.get('score_norm', 0):.2f})"
        )

    uns = state.get("unsupervised_result") or {}
    if uns.get("status") == "ok":
        raw_scores["unsupervised"] = float(uns.get("score_norm", 0.0))
        details["unsupervised"] = (
            f"{uns.get('best_k')} clusters (silhouette={uns.get('silhouette', 0):.2f})"
        )

    rl = state.get("reinforcement_result") or {}
    if rl.get("status") == "ok":
        raw_scores["reinforcement"] = float(rl.get("score_norm", 0.0))
        details["reinforcement"] = (
            f"acierto {rl.get('final_reward', 0)*100:.0f}% vs azar "
            f"{rl.get('baseline', 0)*100:.0f}%"
        )

    if not raw_scores:
        msg = "Ningun analisis produjo resultados comparables; no hay recomendacion."
        bus.warning(AGENT, msg)
        bus.narrate(AGENT, narrate_stream(msg, msg))
        bus.agent_end(AGENT, "Sin recomendacion.")
        return {"recommendation": {"status": "empty", "reason": msg}}

    # Score final ponderado por objetivo de negocio.
    final_scores = {
        k: round(raw_scores[k] * weights.get(k, 0.5), 4) for k in raw_scores
    }
    winner = max(final_scores, key=final_scores.get)

    # Grafico radar de calidad interna (score_norm) por enfoque.
    radar_scores = {APPROACH_LABELS[k]: raw_scores[k] for k in raw_scores}
    bus.chart(AGENT, "Calidad por enfoque", viz.recommendation_radar(radar_scores),
              caption="Calidad interna de cada enfoque (0 a 1), antes de ponderar por objetivo.")

    ranking_txt = "; ".join(
        f"{APPROACH_LABELS[k]}: calidad {raw_scores[k]:.2f} x peso {weights.get(k,0.5):.1f} "
        f"= {final_scores[k]:.2f}"
        for k in sorted(final_scores, key=final_scores.get, reverse=True)
    )

    fallback = (
        f"Tu objetivo es '{goal_labels[business_goal]}'. Segun eso pondero cada enfoque. "
        f"Ranking: {ranking_txt}. "
        f"Recomiendo seguir con {APPROACH_LABELS[winner]} ({details.get(winner, '')}), "
        f"porque combina buena calidad interna con la mayor relevancia para tu objetivo."
    )
    prompt = (
        "Eres el evaluador final. Explica a un principiante por que recomiendas un "
        "enfoque sobre los demas, dejando claro que la eleccion depende de SU objetivo "
        "de negocio (no solo de la metrica mas alta). Se claro y motivador. "
        f"Objetivo: {goal_labels[business_goal]}. Calculo: {ranking_txt}. "
        f"Ganador: {APPROACH_LABELS[winner]} ({details.get(winner,'')})."
    )
    bus.narrate(AGENT, narrate_stream(prompt, fallback, domain=domain))

    recommendation = {
        "status": "ok",
        "business_goal": business_goal,
        "business_goal_label": goal_labels[business_goal],
        "winner": winner,
        "winner_label": APPROACH_LABELS[winner],
        "winner_detail": details.get(winner, ""),
        "raw_scores": raw_scores,
        "weights": {k: weights.get(k, 0.5) for k in raw_scores},
        "final_scores": final_scores,
        "ranking_text": ranking_txt,
        "domain": domain,
    }
    bus.result(AGENT, "recommendation", recommendation)
    bus.agent_end(AGENT, f"Recomendacion: {APPROACH_LABELS[winner]}.")
    return {"recommendation": recommendation}
