"""Capa 7 - Agente de preguntas y respuestas (voz) sobre el estado ya calculado.

Este agente NO entrena modelos ni toca `clean_df` directamente: solo lee
resultados que YA calcularon los agentes anteriores (etl_report, diagnosis,
supervised_result, unsupervised_result, reinforcement_result, recommendation,
executive_report) y responde en lenguaje natural, SIEMPRE citando la fuente
("segun el Agente Supervisado..."). Es un narrador con function-calling sobre
el estado ya existente, nunca un modelo con acceso libre a los datos crudos.

Regla no negociable: cualquier numero que aparezca en la respuesta debe ser
trazable a un numero que ya existe en los hechos extraidos del estado. Esto se
VALIDA automaticamente (`_validate_no_invented_numbers`): si la respuesta del
LLM trae un numero que no esta en los hechos, se descarta y se usa la
plantilla determinista de respaldo, que por construccion solo inserta valores
que ya vienen del estado.
"""

from __future__ import annotations

import re
from typing import Any

from utils.llm import _get_client, narrate_text  # noqa: SLF001 (reuso interno intencional)

AGENT = "AsistenteDeVoz"

QA_SYSTEM_PROMPT = (
    "Eres el asistente de voz del 'Analista de Datos Multiagente'. Respondes "
    "preguntas en espanol sobre un analisis YA realizado. SOLO puedes usar los "
    "hechos que se te entregan explicitamente en el contexto; NUNCA inventes "
    "cifras, porcentajes, nombres de columnas o resultados que no esten ahi. "
    "Si la pregunta no se puede responder con esos hechos, dilo con honestidad "
    "en vez de inventar. SIEMPRE menciona de que agente proviene el dato (ej. "
    "'segun el Agente Supervisado...', 'segun el Agente Recomendador...'). "
    "Responde en 1-3 frases, tono conversacional, como si hablaras en voz alta."
)


# ----------------------------------------------------------------------
# 1. Extraccion de hechos ya calculados (fuente unica de verdad)
# ----------------------------------------------------------------------
def build_context_facts(state: dict[str, Any]) -> dict[str, Any]:
    """Reune SOLO resultados ya calculados por los agentes anteriores."""
    facts: dict[str, Any] = {}

    etl = state.get("etl_report") or {}
    if etl:
        facts["etl"] = {
            "filas": etl.get("overview_after", {}).get("n_rows"),
            "columnas": etl.get("overview_after", {}).get("n_cols"),
            "duplicados_eliminados": etl.get("duplicates_removed"),
            "columnas_imputadas": len(etl.get("imputed", {})),
        }

    diag = state.get("diagnosis") or {}
    if diag:
        facts["diagnostico"] = {
            "variable_objetivo": diag.get("target_column"),
            "enmarcada_como": diag.get("target_frame"),
            "enfoques_aplicables": diag.get("applicable"),
        }

    sup = state.get("supervised_result") or {}
    if sup.get("status") == "ok":
        facts["supervisado"] = {
            "mejor_modelo": sup.get("best_model"),
            "tarea": sup.get("task"),
            "metrica_principal": sup.get("primary_metric"),
            "metricas": sup.get("metrics"),
            "variable_mas_importante": sup.get("top_feature"),
            "score_norm": sup.get("score_norm"),
        }

    uns = state.get("unsupervised_result") or {}
    if uns.get("status") == "ok":
        facts["no_supervisado"] = {
            "numero_de_clusters": uns.get("best_k"),
            "silhouette": uns.get("silhouette"),
            "tamanos_de_cluster": uns.get("cluster_sizes"),
            "rasgos_distintivos": uns.get("distinctive"),
        }

    rl = state.get("reinforcement_result") or {}
    if rl.get("status") == "ok":
        facts["refuerzo"] = {
            "acierto_final": rl.get("final_reward"),
            "acierto_al_azar": rl.get("baseline"),
            "mejora": rl.get("lift"),
        }

    rec = state.get("recommendation") or {}
    if rec.get("status") == "ok":
        facts["recomendacion"] = {
            "enfoque_ganador": rec.get("winner_label"),
            "detalle_ganador": rec.get("winner_detail"),
            "objetivo_de_negocio": rec.get("business_goal_label"),
            "ranking": rec.get("ranking_text"),
        }

    report = state.get("executive_report") or {}
    if report:
        facts["reporte_ejecutivo"] = {
            "titulo": report.get("titulo"),
            "hallazgo_clave": report.get("hallazgo_clave"),
            "siguiente_paso": report.get("siguiente_paso_sugerido"),
            "numero_de_recomendaciones": len(report.get("recomendaciones", [])),
        }

    return facts


# ----------------------------------------------------------------------
# 2. Validacion anti-invencion de cifras
# ----------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _flatten_numbers(obj: Any, out: set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(_norm_num(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM_RE.findall(obj):
            out.add(_norm_num(m))


def _norm_num(value: Any) -> str:
    """Normaliza un numero a una representacion comparable (redondeo laxo)."""
    try:
        f = float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return str(value)
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}"


def _validate_no_invented_numbers(answer: str, facts: dict[str, Any]) -> bool:
    """True si TODO numero en `answer` aparece (redondeado) entre los hechos."""
    allowed: set[str] = set()
    _flatten_numbers(facts, allowed)
    # Toleramos numeros pequenos genericos que no son "datos" (ej. "1-3 frases").
    allowed |= {"1", "2", "3", "0"}

    found = [_norm_num(m) for m in _NUM_RE.findall(answer)]
    for num in found:
        if num not in allowed:
            return False
    return True


# ----------------------------------------------------------------------
# 3. Respuesta determinista (fallback seguro, cubre preguntas frecuentes)
# ----------------------------------------------------------------------
def _deterministic_answer(question: str, facts: dict[str, Any]) -> tuple[str, str]:
    """Responde con plantillas que SOLO insertan valores ya presentes en facts.

    Devuelve (respuesta, agente_fuente).
    """
    q = question.lower()

    if any(k in q for k in ("mejor modelo", "mejor enfoque", "que conviene",
                            "recomendacion", "recomendación")):
        rec = facts.get("recomendacion")
        if rec:
            return (
                f"Segun el Agente Recomendador, el enfoque que conviene seguir es "
                f"{rec['enfoque_ganador']} ({rec['detalle_ganador']}), para tu "
                f"objetivo de {rec['objetivo_de_negocio']}.",
                "Recomendador",
            )

    if any(k in q for k in ("variable", "factor", "importa")):
        sup = facts.get("supervisado")
        if sup and sup.get("variable_mas_importante"):
            return (
                f"Segun el Agente Supervisado, la variable que mas influye es "
                f"'{sup['variable_mas_importante']}', usando el modelo "
                f"{sup['mejor_modelo']}.",
                "Supervisado",
            )

    if any(k in q for k in ("cuantas filas", "cuantos registros", "tamano del dataset",
                            "cuántas filas", "cuántos registros")):
        etl = facts.get("etl")
        if etl:
            return (
                f"Segun el Agente ETL, el dataset limpio tiene {etl['filas']} filas "
                f"y {etl['columnas']} columnas.",
                "ETL",
            )

    if any(k in q for k in ("cluster", "grupo", "segmento")):
        uns = facts.get("no_supervisado")
        if uns:
            return (
                f"Segun el Agente No Supervisado, se encontraron "
                f"{uns['numero_de_clusters']} grupos, con un silhouette de "
                f"{uns['silhouette']:.2f}.",
                "NoSupervisado",
            )

    if any(k in q for k in ("refuerzo", "bandit", "decision")):
        rl = facts.get("refuerzo")
        if rl:
            return (
                f"Segun el Agente de Refuerzo, la politica aprendida acierta "
                f"{rl['acierto_final']*100:.0f}% de las veces, frente a "
                f"{rl['acierto_al_azar']*100:.0f}% de elegir al azar.",
                "Refuerzo",
            )

    if any(k in q for k in ("siguiente paso", "que hago ahora", "que sigue")):
        rep = facts.get("reporte_ejecutivo")
        if rep:
            return (
                f"Segun el Reporte Ejecutivo, el siguiente paso sugerido es: "
                f"{rep['siguiente_paso']}",
                "ReporteEjecutivo",
            )

    if any(k in q for k in ("hallazgo", "que encontraron", "que se descubrio")):
        rep = facts.get("reporte_ejecutivo")
        if rep:
            return (
                f"Segun el Reporte Ejecutivo, el hallazgo clave es: "
                f"{rep['hallazgo_clave']}",
                "ReporteEjecutivo",
            )

    # Sin coincidencia: resumen general honesto, sin inventar nada.
    disponibles = ", ".join(facts.keys()) or "ningun resultado todavia"
    return (
        f"No tengo un dato exacto para esa pregunta en lo que ya calculamos. "
        f"Puedo hablarte de: {disponibles}. Intenta preguntar, por ejemplo, "
        f"'cual fue el mejor modelo' o 'que variable importa mas'.",
        "AsistenteDeVoz",
    )


# ----------------------------------------------------------------------
# 4. Entrada publica: responde una pregunta usando SOLO el estado
# ----------------------------------------------------------------------
def answer_question(question: str, state: dict[str, Any]) -> dict[str, Any]:
    """Responde `question` usando unicamente resultados ya calculados en `state`.

    Devuelve {"answer": str, "source_agent": str, "engine": "llm"|"template"}.
    """
    facts = build_context_facts(state)
    fallback_answer, fallback_source = _deterministic_answer(question, facts)

    client = _get_client()
    if client is None or not facts:
        return {"answer": fallback_answer, "source_agent": fallback_source,
                "engine": "template", "facts_used": facts}

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Hechos disponibles (unica fuente permitida, en espanol, formato "
        f"clave:valor por agente):\n{facts}\n\n"
        f"Responde la pregunta usando SOLO estos hechos. Si no hay dato "
        f"suficiente, dilo. Cita el agente fuente."
    )
    try:
        llm_answer = narrate_text(prompt, fallback_text=fallback_answer)
    except Exception:
        llm_answer = fallback_answer

    if llm_answer and _validate_no_invented_numbers(llm_answer, facts):
        return {"answer": llm_answer, "source_agent": "LLM (validado contra el estado)",
                "engine": "llm", "facts_used": facts}

    # La respuesta del LLM traia numeros no trazables: descartamos y usamos
    # la plantilla determinista, que por construccion nunca inventa cifras.
    return {"answer": fallback_answer, "source_agent": fallback_source,
            "engine": "template_validated_fallback", "facts_used": facts}
