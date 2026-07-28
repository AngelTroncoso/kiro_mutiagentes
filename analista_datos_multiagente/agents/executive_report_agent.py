"""Capa 6 - Agente de Reporte Ejecutivo Gerencial.

Se ejecuta al final del pipeline, despues del Recomendador. Traduce los hallazgos
tecnicos (diagnostico, resultados de los tres enfoques y recomendacion normalizada)
en un INFORME GERENCIAL para quien toma decisiones de negocio, con tono ejecutivo.

Principio rector (igual que el resto de la app): NUNCA inventar cifras. Todos los
numeros salen de `clean_df`, `diagnosis` o las metricas ya calculadas. El agente:

    1. Extrae "hechos" trazables de los datos (universo, distribucion del objetivo,
       metricas, ganador).
    2. Construye un reporte DETERMINISTA completo a partir de esos hechos (esta es
       la fuente de verdad de las cifras y el fallback si no hay LLM).
    3. Si hay LLM, le pide reescribir SOLO el texto con tono ejecutivo, pasandole los
       hechos y prohibiendole cambiar numeros. Si la respuesta no es valida (no trae
       exactamente 3 recomendaciones), se conserva el reporte determinista.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from utils.domain_config import frame_target, get_domain
from utils.llm import complete_json, executive_system_prompt_for

AGENT = "ReporteEjecutivo"

# Estructura vacia de una recomendacion, para validar la salida del LLM.
_REC_FIELDS = {"titulo", "descripcion", "impacto_esperado", "riesgo_o_consideracion"}


# ----------------------------------------------------------------------
# 1. Extraccion de hechos trazables
# ----------------------------------------------------------------------
def _extract_facts(state) -> dict[str, Any]:
    df: pd.DataFrame = state.get("clean_df")
    diagnosis = state.get("diagnosis", {}) or {}
    rec = state.get("recommendation", {}) or {}
    target = state.get("target_column") or diagnosis.get("target_column")
    domain = state.get("domain", "general")
    domain_cfg = get_domain(domain)

    facts: dict[str, Any] = {
        "n_rows": int(df.shape[0]) if df is not None else 0,
        "n_cols": int(df.shape[1]) if df is not None else 0,
        "target": target,
        # Enmarca el NOMBRE de la columna objetivo con lenguaje de dominio; no
        # cambia cual es el objetivo ni inventa ninguna columna o cifra nueva.
        "target_frame": diagnosis.get("target_frame") or frame_target(target, domain),
        "domain": domain,
        "domain_label": domain_cfg["label"],
        "kpi_context": domain_cfg["kpi_context"],
        "business_goal": rec.get("business_goal", state.get("business_goal")),
        "business_goal_label": rec.get("business_goal_label", ""),
        "winner": rec.get("winner"),
        "winner_label": rec.get("winner_label", ""),
        "winner_detail": rec.get("winner_detail", ""),
        "ranking_text": rec.get("ranking_text", ""),
    }

    # --- Distribucion del objetivo (cifra concreta del hallazgo) ---
    facts["target_kind"] = None
    facts["target_distribution"] = {}
    if df is not None and target in df.columns:
        serie = df[target]
        sup = state.get("supervised_result") or {}
        task = sup.get("task") or (diagnosis.get("supervised", {}) or {}).get("task")
        if task == "regresion" or (pd.api.types.is_numeric_dtype(serie)
                                   and serie.nunique() > 20):
            facts["target_kind"] = "continua"
            facts["target_distribution"] = {
                "media": round(float(serie.mean()), 2),
                "mediana": round(float(serie.median()), 2),
                "min": round(float(serie.min()), 2),
                "max": round(float(serie.max()), 2),
            }
        else:
            facts["target_kind"] = "categorica"
            counts = serie.value_counts()
            total = int(counts.sum())
            dist = {str(k): {"n": int(v), "pct": round(100.0 * v / total, 1)}
                    for k, v in counts.head(6).items()}
            facts["target_distribution"] = dist
            # Clase mayoritaria y minoritaria (utiles para el hallazgo).
            facts["clase_mayoritaria"] = {"valor": str(counts.idxmax()),
                                          "pct": round(100.0 * counts.max() / total, 1)}
            facts["clase_minoritaria"] = {"valor": str(counts.idxmin()),
                                          "pct": round(100.0 * counts.min() / total, 1)}

    # --- Metricas por enfoque (solo las ejecutadas con exito) ---
    sup = state.get("supervised_result") or {}
    if sup.get("status") == "ok":
        facts["supervised"] = {
            "task": sup.get("task"),
            "primary_metric": sup.get("primary_metric"),
            "metrics": sup.get("metrics", {}),
            "top_feature": sup.get("top_feature"),
            "score_norm": round(float(sup.get("score_norm", 0)), 3),
        }
    uns = state.get("unsupervised_result") or {}
    if uns.get("status") == "ok":
        facts["unsupervised"] = {
            "best_k": uns.get("best_k"),
            "silhouette": round(float(uns.get("silhouette", 0)), 3),
            "cluster_sizes": uns.get("cluster_sizes", {}),
            "distinctive": uns.get("distinctive", ""),
        }
    rl = state.get("reinforcement_result") or {}
    if rl.get("status") == "ok":
        facts["reinforcement"] = {
            "final_reward": round(float(rl.get("final_reward", 0)), 3),
            "baseline": round(float(rl.get("baseline", 0)), 3),
            "lift": round(float(rl.get("lift", 0)), 3),
        }
    return facts


# ----------------------------------------------------------------------
# 2. Reporte determinista (fuente de verdad + fallback)
# ----------------------------------------------------------------------
def _fmt_target(target: str | None) -> str:
    if not target:
        return "la variable de interes"
    return str(target).replace("_", " ")


def _hallazgo(facts: dict[str, Any]) -> str:
    target = _fmt_target(facts.get("target"))
    frame = facts.get("target_frame")
    frame_note = f" (marco de referencia: {frame})" if frame else ""
    kind = facts.get("target_kind")
    if kind == "categorica" and facts.get("clase_mayoritaria"):
        may = facts["clase_mayoritaria"]
        mino = facts.get("clase_minoritaria", {})
        base = (f"En '{target}'{frame_note}, el valor mas frecuente es '{may['valor']}' con "
                f"{may['pct']}% de los {facts['n_rows']} registros analizados.")
        if mino and mino.get("valor") != may.get("valor"):
            base += (f" El grupo menos representado ('{mino['valor']}') concentra "
                     f"solo {mino['pct']}%, lo que marca un desbalance a vigilar.")
        return base
    if kind == "continua" and facts.get("target_distribution"):
        d = facts["target_distribution"]
        return (f"En '{target}'{frame_note}, el valor promedio es {d['media']} "
                f"(mediana {d['mediana']}), con un rango que va de {d['min']} a "
                f"{d['max']} en los {facts['n_rows']} registros analizados.")
    return (f"Se analizaron {facts['n_rows']} registros sobre '{target}'{frame_note}; "
            f"los datos quedaron completos y listos para decidir.")


def _rec_supervised(facts: dict[str, Any]) -> list[dict[str, str]]:
    target = _fmt_target(facts.get("target"))
    sup = facts.get("supervised", {})
    metrics = sup.get("metrics", {})
    top = sup.get("top_feature", "el factor mas influyente")
    top_fmt = str(top).replace("_", " ")
    if sup.get("task") == "clasificacion":
        acc = metrics.get("accuracy")
        perf = f"{acc*100:.0f}% de aciertos" if acc is not None else "un desempeno solido"
        impacto1 = (f"Anticipar casos de '{target}' con {perf} permite actuar antes de "
                    f"que ocurran, en lugar de reaccionar tarde.")
    else:
        r2 = metrics.get("r2")
        perf = (f"explicando el {r2*100:.0f}% de la variacion" if r2 is not None
                else "con buena precision")
        impacto1 = (f"Estimar '{target}' por adelantado {perf} da margen para planificar "
                    f"con datos y no por intuicion.")
    return [
        {
            "titulo": f"Anticiparse usando '{target}' como senal temprana",
            "descripcion": (f"Incorporar la estimacion de '{target}' al proceso de decision "
                            f"para priorizar los casos antes de que se materialicen."),
            "impacto_esperado": impacto1,
            "riesgo_o_consideracion": ("La estimacion se basa en los datos historicos "
                                       "disponibles; conviene revalidarla al cambiar el contexto."),
        },
        {
            "titulo": f"Focalizar recursos en '{top_fmt}'",
            "descripcion": (f"'{top_fmt}' es el factor que mas incide en '{target}'. "
                            f"Concentrar esfuerzo y presupuesto donde este factor pesa mas."),
            "impacto_esperado": ("Mayor retorno por cada peso invertido al actuar sobre la "
                                 "palanca de mayor efecto en lugar de repartir sin foco."),
            "riesgo_o_consideracion": ("Correlacion no implica causa: conviene confirmar la "
                                       "relacion con una prueba controlada antes de escalar."),
        },
        {
            "titulo": "Institucionalizar el seguimiento del indicador",
            "descripcion": (f"Definir un tablero mensual de '{target}' y sus factores clave, "
                            f"con responsables y umbrales de alerta."),
            "impacto_esperado": ("Deteccion temprana de desvios y decisiones mas rapidas, "
                                 "sostenidas en el tiempo."),
            "riesgo_o_consideracion": ("Requiere datos frescos y de calidad de forma "
                                       "recurrente; sin eso, el seguimiento pierde valor."),
        },
    ]


def _rec_unsupervised(facts: dict[str, Any]) -> list[dict[str, str]]:
    uns = facts.get("unsupervised", {})
    k = uns.get("best_k", "varios")
    distinctive = uns.get("distinctive", "")
    return [
        {
            "titulo": f"Adoptar una estrategia segmentada en {k} grupos",
            "descripcion": (f"Los datos revelan {k} segmentos naturales con perfiles "
                            f"distintos. Disenar propuestas diferenciadas para cada uno."),
            "impacto_esperado": ("Mensajes y ofertas mas relevantes por segmento, con mejor "
                                 "conversion que un enfoque unico para todos."),
            "riesgo_o_consideracion": ("Los segmentos describen patrones actuales; conviene "
                                       "revisarlos periodicamente porque pueden evolucionar."),
        },
        {
            "titulo": "Priorizar el segmento de mayor valor",
            "descripcion": ("Identificar cual de los grupos aporta mas valor de negocio y "
                            "asignarle atencion preferente."
                            + (f" Rasgos que distinguen los grupos: {distinctive}."
                               if distinctive else "")),
            "impacto_esperado": ("Uso mas eficiente de recursos comerciales al concentrarlos "
                                 "donde el retorno es mayor."),
            "riesgo_o_consideracion": ("Definir 'valor' con criterio de negocio explicito para "
                                       "no sesgar la priorizacion."),
        },
        {
            "titulo": "Validar los segmentos con las areas operativas",
            "descripcion": ("Contrastar los grupos hallados con el conocimiento de ventas y "
                            "operaciones antes de accionarlos."),
            "impacto_esperado": ("Segmentacion accionable y creible internamente, con mayor "
                                 "adopcion por los equipos."),
            "riesgo_o_consideracion": ("Si un grupo no tiene interpretacion de negocio clara, "
                                       "no conviene forzar acciones sobre el."),
        },
    ]


def _rec_reinforcement(facts: dict[str, Any]) -> list[dict[str, str]]:
    target = _fmt_target(facts.get("target"))
    rl = facts.get("reinforcement", {})
    fr = rl.get("final_reward")
    base = rl.get("baseline")
    perf = (f"acierta {fr*100:.0f}% de las decisiones frente a {base*100:.0f}% al azar"
            if fr is not None and base is not None else "supera a la decision al azar")
    return [
        {
            "titulo": "Pilotar una politica de decision guiada por datos",
            "descripcion": (f"Probar en un piloto acotado una regla que elija la mejor accion "
                            f"sobre '{target}' segun el perfil de cada caso."),
            "impacto_esperado": (f"La estrategia {perf}, lo que sugiere margen real de mejora "
                                 f"frente a decidir sin datos."),
            "riesgo_o_consideracion": ("Es una simulacion sobre datos historicos, no un entorno "
                                       "real; el piloto debe confirmar el resultado en vivo."),
        },
        {
            "titulo": "Medir el impacto con un grupo de control",
            "descripcion": ("Ejecutar el piloto contra un grupo de control para aislar el "
                            "efecto real de la nueva politica de decision."),
            "impacto_esperado": ("Evidencia solida del beneficio incremental antes de invertir "
                                 "en un despliegue amplio."),
            "riesgo_o_consideracion": ("Requiere disciplina en la medicion; sin control, la "
                                       "mejora observada puede deberse a otros factores."),
        },
        {
            "titulo": "Escalar por etapas segun resultados",
            "descripcion": ("Ampliar la politica solo si el piloto confirma la mejora, "
                            "avanzando por fases controladas."),
            "impacto_esperado": ("Captura del beneficio limitando el riesgo de un despliegue "
                                 "prematuro."),
            "riesgo_o_consideracion": ("Definir de antemano los criterios de exito que habilitan "
                                       "pasar de una etapa a la siguiente."),
        },
    ]


def _build_deterministic_report(facts: dict[str, Any]) -> dict[str, Any]:
    target = _fmt_target(facts.get("target"))
    winner = facts.get("winner")
    goal_label = facts.get("business_goal_label") or "tomar mejores decisiones"
    domain_label = facts.get("domain_label")

    if winner == "unsupervised":
        recs = _rec_unsupervised(facts)
    elif winner == "reinforcement":
        recs = _rec_reinforcement(facts)
    else:  # supervised o por defecto
        recs = _rec_supervised(facts)

    domain_clause = f" desde una perspectiva de {domain_label}" if domain_label else ""
    resumen = (f"Se analizo{domain_clause} un universo de {facts['n_rows']} registros "
               f"con {facts['n_cols']} variables, enfocado en '{target}'. "
               f"El objetivo de negocio planteado fue: {goal_label.lower()}. "
               f"El analisis apunta a un enfoque de "
               f"{facts.get('winner_label', 'analisis de datos').lower()}.")

    siguiente = (f"Aprobar un piloto acotado alineado a '{target}' y asignar un responsable "
                 f"para ejecutar la primera recomendacion en el proximo ciclo.")

    return {
        "titulo": f"Reporte Ejecutivo: Analisis de {target.title()}",
        "resumen_situacion": resumen,
        "hallazgo_clave": _hallazgo(facts),
        "recomendaciones": recs,
        "siguiente_paso_sugerido": siguiente,
    }


# ----------------------------------------------------------------------
# 3. Validacion y enriquecimiento con LLM
# ----------------------------------------------------------------------
def _is_valid_report(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for key in ("titulo", "resumen_situacion", "hallazgo_clave",
                "recomendaciones", "siguiente_paso_sugerido"):
        if key not in obj:
            return False
    recs = obj.get("recomendaciones")
    if not isinstance(recs, list) or len(recs) != 3:
        return False
    for r in recs:
        if not isinstance(r, dict) or not _REC_FIELDS.issubset(r.keys()):
            return False
    return True


def _enrich_with_llm(facts: dict[str, Any], base: dict[str, Any],
                     domain: str) -> dict[str, Any]:
    domain_label = facts.get("domain_label", "")
    kpi_context = facts.get("kpi_context", "")
    domain_note = (
        f" Estas escribiendo para el area de {domain_label}; usa el marco de "
        f"referencia de {kpi_context} donde sea natural, SIN inventar ningun ratio "
        f"o cifra que no este en los hechos."
        if domain_label else ""
    )
    prompt = (
        "Redacta un reporte ejecutivo en espanol, tono gerencial, a partir UNICAMENTE "
        "de los siguientes hechos (no agregues ni inventes ninguna cifra). Devuelve un "
        "objeto JSON con EXACTAMENTE estas claves: 'titulo' (string), "
        "'resumen_situacion' (string, 2-4 frases), 'hallazgo_clave' (string con una "
        "cifra concreta tomada de los hechos), 'recomendaciones' (lista de EXACTAMENTE "
        "3 objetos, cada uno con 'titulo', 'descripcion', 'impacto_esperado', "
        "'riesgo_o_consideracion'), y 'siguiente_paso_sugerido' (string de una linea). "
        "Las 3 recomendaciones deben ser accionables por una gerencia (donde enfocar "
        "presupuesto, que investigar, que monitorear), distintas entre si, y coherentes "
        f"con el enfoque recomendado y el objetivo de negocio. Nada de jerga tecnica."
        f"{domain_note}\n\n"
        f"HECHOS (fuente unica de datos):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"BORRADOR DE REFERENCIA (puedes mejorar la redaccion, respetando las cifras):\n"
        f"{json.dumps(base, ensure_ascii=False)}"
    )
    result = complete_json(prompt, fallback_obj=base,
                           system_prompt=executive_system_prompt_for(domain))
    return result if _is_valid_report(result) else base


# ----------------------------------------------------------------------
# Entrada del agente
# ----------------------------------------------------------------------
def run_executive_report(state) -> dict[str, Any]:
    bus = state.get("bus")
    if bus is not None:
        bus.agent_start(AGENT, "Reporte Ejecutivo - traduccion a decision de negocio",
                        icon="📄")

    domain = state.get("domain", "general")
    facts = _extract_facts(state)
    base = _build_deterministic_report(facts)
    report = _enrich_with_llm(facts, base, domain)

    # Metadatos utiles para trazabilidad y la UI.
    report["_meta"] = {
        "n_rows": facts["n_rows"],
        "n_cols": facts["n_cols"],
        "target": facts.get("target"),
        "winner": facts.get("winner"),
        "winner_label": facts.get("winner_label"),
        "business_goal_label": facts.get("business_goal_label"),
        "domain": domain,
        "domain_label": facts.get("domain_label"),
    }

    if bus is not None:
        bus.result(AGENT, "executive_report", report)
        bus.agent_end(AGENT, f"Reporte ejecutivo generado: {report.get('titulo','')}.")

    return {"executive_report": report}
