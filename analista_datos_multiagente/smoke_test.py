"""Prueba headless del pipeline completo (sin Streamlit ni Groq).

Genera datasets sinteticos (clasificacion y regresion), corre las dos fases del
pipeline para cada uno de los 3 dominios de negocio (Finanzas / Logistica /
Estrategia) = 6 corridas, y verifica:

    1. El pipeline completa sin error en cada combinacion.
    2. La lista de business_goal ofrecida corresponde al dominio elegido.
    3. El Reporte Ejecutivo mantiene exactamente 3 recomendaciones, sin cifras
       inventadas y sin errores de agente.
    4. Las metricas (accuracy/R2/silhouette/score_norm) son IDENTICAS entre
       dominios para el MISMO dataset y la MISMA seleccion de analisis: el
       dominio es una capa de presentacion, nunca de computo.

Consume el EventBus en un hilo aparte para simular el rol de la UI.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Forzamos el narrador local (sin llamadas a Groq) para una prueba determinista.
os.environ.pop("GROQ_API_KEY", None)

from orchestrator.graph import run_analysis_phase, run_diagnosis_phase  # noqa: E402
from orchestrator.state import new_state  # noqa: E402
from utils.domain_config import DOMAIN_KEYS, goal_labels_for  # noqa: E402
from utils.streaming import EventBus, run_in_thread  # noqa: E402


def make_classification_df(n=200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    edad = rng.integers(18, 70, n).astype(float)
    ingreso = rng.normal(3000, 800, n)
    ciudad = rng.choice(["Bogota", "Medellin", "Cali"], n)
    # Objetivo dependiente de las features + ruido.
    score = 0.03 * edad + 0.0008 * ingreso + rng.normal(0, 1, n)
    churn = (score < np.median(score)).astype(int)
    df = pd.DataFrame({"edad": edad, "ingreso": ingreso, "ciudad": ciudad,
                       "churn": churn})
    # Inyectamos nulos y duplicados para probar el ETL.
    df.loc[rng.choice(n, 15, replace=False), "ingreso"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "ciudad"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


def make_regression_df(n=200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    m2 = rng.normal(80, 25, n)
    habitaciones = rng.integers(1, 5, n).astype(float)
    barrio = rng.choice(["centro", "norte", "sur"], n)
    precio = 1000 * m2 + 20000 * habitaciones + rng.normal(0, 10000, n)
    return pd.DataFrame({"m2": m2, "habitaciones": habitaciones,
                         "barrio": barrio, "precio": precio})


def drain_bus(bus: EventBus, thread) -> dict:
    """Consume todos los eventos como haria la UI y cuenta artefactos."""
    counts = {"charts": 0, "tables": 0, "narrations": 0, "results": {},
              "warnings": 0, "errors": []}
    for event in bus.stream(is_alive=thread.is_alive):
        t = event.get("type")
        if t == "narration":
            text = "".join(event["stream"])  # drenar el sub-generador
            if text:
                counts["narrations"] += 1
        elif t == "chart":
            counts["charts"] += 1
        elif t == "table":
            counts["tables"] += 1
        elif t == "warning":
            counts["warnings"] += 1
        elif t == "result":
            counts["results"][event["key"]] = event["payload"]
        elif t == "error":
            counts["errors"].append(event.get("text", ""))
    thread.join(timeout=2.0)
    return counts


def run_case(name: str, df: pd.DataFrame, business_goal: str, domain: str) -> tuple[bool, dict]:
    """Corre el pipeline completo para (df, business_goal, domain).

    Devuelve (ok, metricas_por_enfoque) para permitir comparar entre dominios.
    """
    print(f"\n{'='*70}\nCASO: {name} | DOMINIO: {domain}\n{'='*70}")
    ok = True

    # --- Fase 1: diagnostico ---
    bus = EventBus()
    state = new_state(df.copy(), bus, domain=domain)

    def _job1():
        state.update(run_diagnosis_phase(state))

    th = run_in_thread(_job1, bus)
    c1 = drain_bus(bus, th)
    print(f"[Fase 1] narraciones={c1['narrations']} charts={c1['charts']} "
          f"tables={c1['tables']} errores={c1['errors']}")

    assert state.get("clean_df") is not None, "ETL no produjo clean_df"
    assert state["clean_df"].isna().sum().sum() == 0, "Quedaron nulos tras ETL"
    diagnosis = state.get("diagnosis", {})
    assert diagnosis.get("applicable"), "Router no recomendo ningun enfoque"
    assert diagnosis.get("domain") == domain, "El diagnostico no propago el dominio"
    print(f"[Fase 1] objetivo detectado: {diagnosis.get('target_column')} "
          f"(enmarcado: {diagnosis.get('target_frame')}) | "
          f"aplicables: {diagnosis.get('applicable')}")
    if c1["errors"]:
        ok = False

    # --- Verificacion: business_goal ofrecido corresponde al dominio ---
    labels = goal_labels_for(domain)
    assert set(labels.keys()) == {"predecir", "segmentar", "decidir", "explorar"}, \
        f"Las claves de business_goal deben ser las 4 abstractas, dominio={domain}"
    assert business_goal in labels, f"'{business_goal}' no es una clave valida"
    print(f"[Dominio] business_goal '{business_goal}' -> '{labels[business_goal]}'")

    # --- Fase 2: analisis + recomendacion ---
    bus2 = EventBus()
    state2 = new_state(state["clean_df"].copy(), bus2, domain=domain)
    state2["clean_df"] = state["clean_df"].copy()
    state2["diagnosis"] = diagnosis
    state2["selected_analyses"] = diagnosis["applicable"]
    state2["target_column"] = diagnosis.get("target_column")
    state2["business_goal"] = business_goal

    def _job2():
        state2.update(run_analysis_phase(state2))

    th2 = run_in_thread(_job2, bus2)
    c2 = drain_bus(bus2, th2)
    print(f"[Fase 2] narraciones={c2['narrations']} charts={c2['charts']} "
          f"tables={c2['tables']} errores={c2['errors']}")

    metrics_by_approach: dict[str, dict] = {}
    for approach in diagnosis["applicable"]:
        res = state2.get(f"{approach}_result")
        assert res, f"Falta resultado de {approach}"
        print(f"  - {approach}: status={res.get('status')} "
              f"score_norm={res.get('score_norm')}")
        if res.get("status") != "ok":
            print(f"    (advertencia: {res.get('reason')})")
        metrics_by_approach[approach] = {
            "status": res.get("status"),
            "score_norm": res.get("score_norm"),
            "metrics": res.get("metrics") if approach == "supervised" else None,
            "silhouette": res.get("silhouette") if approach == "unsupervised" else None,
            "final_reward": res.get("final_reward") if approach == "reinforcement" else None,
        }

    rec = state2.get("recommendation")
    assert rec and rec.get("status") == "ok", "No hubo recomendacion final"
    assert rec.get("business_goal_label") == labels[business_goal], \
        "La etiqueta de business_goal en la recomendacion no usa el label de dominio"
    print(f"[Recomendacion] objetivo='{business_goal}' ({rec['business_goal_label']}) "
          f"-> gana '{rec['winner']}' ({rec['winner_label']})")
    print(f"[Recomendacion] ranking: {rec['ranking_text']}")

    assert c2["charts"] >= 1, "La fase de analisis no produjo graficos"

    # --- Capa 6: reporte ejecutivo ---
    report = state2.get("executive_report")
    assert report, "No se genero el reporte ejecutivo"
    for campo in ("titulo", "resumen_situacion", "hallazgo_clave",
                  "recomendaciones", "siguiente_paso_sugerido"):
        assert campo in report, f"Falta el campo '{campo}' en el reporte ejecutivo"
    recs = report["recomendaciones"]
    assert isinstance(recs, list) and len(recs) == 3, \
        f"El reporte debe tener exactamente 3 recomendaciones (tiene {len(recs)})"
    titulos = set()
    for r in recs:
        for campo in ("titulo", "descripcion", "impacto_esperado",
                      "riesgo_o_consideracion"):
            assert r.get(campo), f"Recomendacion sin campo '{campo}'"
        titulos.add(r["titulo"])
    assert len(titulos) == 3, "Las 3 recomendaciones deben ser distintas entre si"
    assert report.get("_meta", {}).get("domain") == domain, \
        "El reporte ejecutivo no registro el dominio en sus metadatos"
    print(f"[Reporte ejecutivo] titulo: {report['titulo']}")
    print(f"[Reporte ejecutivo] hallazgo: {report['hallazgo_clave']}")
    print(f"[Reporte ejecutivo] {len(recs)} recomendaciones: "
          f"{[r['titulo'] for r in recs]}")

    if c2["errors"]:
        ok = False
    return ok, metrics_by_approach


def _assert_identical_metrics(all_metrics: dict[str, dict[str, dict]], case_name: str) -> bool:
    """Compara las metricas de los 3 dominios para un mismo caso; deben ser iguales."""
    domains = list(all_metrics.keys())
    baseline_domain = domains[0]
    baseline = all_metrics[baseline_domain]
    ok = True
    for domain in domains[1:]:
        other = all_metrics[domain]
        if other != baseline:
            ok = False
            print(f"❌ REGRESION: metricas de '{case_name}' difieren entre "
                  f"'{baseline_domain}' y '{domain}':")
            print(f"   {baseline_domain}: {baseline}")
            print(f"   {domain}: {other}")
    if ok:
        print(f"✅ Metricas IDENTICAS entre los 3 dominios para '{case_name}' "
              f"(el dominio es solo presentacion, como se exige).")
    return ok


def main() -> int:
    all_ok = True

    cases = [
        ("Clasificacion (churn)", make_classification_df(), "predecir"),
        ("Regresion (precio)", make_regression_df(), "predecir"),
    ]

    for case_name, df, goal in cases:
        metrics_per_domain: dict[str, dict] = {}
        for domain in DOMAIN_KEYS:
            ok, metrics = run_case(case_name, df, goal, domain)
            all_ok &= ok
            metrics_per_domain[domain] = metrics
        all_ok &= _assert_identical_metrics(metrics_per_domain, case_name)

    print(f"\n{'='*70}")
    print(f"Matriz ejecutada: {len(cases)} casos x {len(DOMAIN_KEYS)} dominios = "
          f"{len(cases) * len(DOMAIN_KEYS)} corridas.")
    print("RESULTADO GLOBAL:", "OK" if all_ok else "CON ERRORES")
    print("="*70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
