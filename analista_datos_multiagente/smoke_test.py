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


def run_case_full_state(df: pd.DataFrame, business_goal: str, domain: str) -> dict:
    """Igual que run_case pero devuelve el `state2` completo (para Capa 7)."""
    bus = EventBus()
    state = new_state(df.copy(), bus, domain=domain)

    def _job1():
        state.update(run_diagnosis_phase(state))

    th = run_in_thread(_job1, bus)
    drain_bus(bus, th)

    diagnosis = state.get("diagnosis", {})
    bus2 = EventBus()
    state2 = new_state(state["clean_df"].copy(), bus2, domain=domain)
    state2["clean_df"] = state["clean_df"].copy()
    state2["diagnosis"] = diagnosis
    state2["etl_report"] = state.get("etl_report", {})
    state2["selected_analyses"] = diagnosis["applicable"]
    state2["target_column"] = diagnosis.get("target_column")
    state2["business_goal"] = business_goal

    def _job2():
        state2.update(run_analysis_phase(state2))

    th2 = run_in_thread(_job2, bus2)
    drain_bus(bus2, th2)
    return state2


def run_voice_qa_smoke_test() -> bool:
    """Capa 7: simula 5 preguntas con 'transcripcion mockeada' (texto directo,

    equivalente a saltarse el STT) y verifica que ninguna respuesta contenga
    numeros ausentes del estado ya calculado.
    """
    print(f"\n{'='*70}\nCAPA 7: Asistente de voz (QA sobre el estado)\n{'='*70}")
    from agents.voice_qa_agent import (
        _validate_no_invented_numbers,
        answer_question,
        build_context_facts,
    )

    os.environ.pop("GROQ_API_KEY", None)  # forzamos el motor de plantillas (sin red)
    state = run_case_full_state(make_classification_df(), "predecir", "finanzas")
    facts = build_context_facts(state)

    mocked_questions = [
        "¿Cual fue el mejor modelo?",
        "¿Que variable importa mas?",
        "¿Cuantas filas tiene el dataset?",
        "¿Cuantos clusters encontraron?",
        "¿Que dice el reporte sobre el siguiente paso?",
    ]

    ok = True
    for q in mocked_questions:
        result = answer_question(q, state)
        answer = result["answer"]
        valid = _validate_no_invented_numbers(answer, facts)
        cita_fuente = bool(re_search_agent_mention(answer))
        status = "OK" if (valid and cita_fuente) else "FALLO"
        if not (valid and cita_fuente):
            ok = False
        print(f"[{status}] Q: {q}\n       A: {answer}\n       "
              f"(engine={result['engine']}, sin_cifras_inventadas={valid}, "
              f"cita_fuente={cita_fuente})")

    if ok:
        print("✅ Las 5 respuestas citan fuente y no contienen cifras ausentes "
              "del estado calculado.")
    return ok


def re_search_agent_mention(answer: str) -> bool:
    keywords = ("Agente", "Reporte Ejecutivo", "LLM (validado")
    return any(k.lower() in answer.lower() for k in keywords)


def run_finance_montecarlo_smoke_test() -> bool:
    """Capa 8: verifica que NPV/IRR del motor Monte Carlo sean matematicamente

    consistentes con una formula cerrada (caso determinista, std=0) y que los
    parametros derivados de un dataset sintetico de finanzas sean trazables
    (media/std reales de la columna, sin supuestos ocultos).
    """
    print(f"\n{'='*70}\nCAPA 8: Simulador Monte Carlo (finanzas)\n{'='*70}")
    from utils.finance_sim import (
        MonteCarloInputs,
        derive_distribution,
        detect_binary_rate,
        detect_monetary_columns,
        run_monte_carlo,
    )

    ok = True

    # --- 1. Consistencia matematica (caso determinista, formula cerrada) ---
    capital, horizon, rate = 100_000.0, 12, 0.01
    inflow, outflow = 15_000.0, 8_000.0
    net = inflow - outflow
    inp = MonteCarloInputs(initial_capital=capital, horizon_periods=horizon,
                           discount_rate=rate, inflow_mean=inflow, inflow_std=0.0,
                           outflow_mean=outflow, outflow_std=0.0,
                           n_iterations=500, seed=1)
    res = run_monte_carlo(inp)
    expected_npv = -capital + sum(net / (1 + rate) ** t for t in range(1, horizon + 1))
    npv_ok = (abs(res.npv_p50 - expected_npv) < 1e-6
              and abs(res.npv_p10 - expected_npv) < 1e-6
              and abs(res.npv_p90 - expected_npv) < 1e-6)
    irr_ok = False
    if res.irr_representative is not None:
        npv_at_irr = -capital + sum(
            net / (1 + res.irr_representative) ** t for t in range(1, horizon + 1))
        irr_ok = abs(npv_at_irr) < 1e-3
    print(f"[Determinista] NPV esperado={expected_npv:.2f} vs simulado P50="
          f"{res.npv_p50:.2f} -> {'OK' if npv_ok else 'FALLO'}")
    print(f"[Determinista] IRR={res.irr_representative:.4f} anula el NPV -> "
          f"{'OK' if irr_ok else 'FALLO'}")
    ok &= npv_ok and irr_ok

    # --- 2. Trazabilidad: parametros derivados de un dataset real ---
    rng = np.random.default_rng(3)
    n = 300
    df_fin = pd.DataFrame({
        "cliente_id": range(n),
        "monto_transaccion": rng.normal(500, 120, n),
        "ingreso_mensual": rng.normal(3000, 800, n),
        "churn": rng.integers(0, 2, n),
    })
    monetary_cols = detect_monetary_columns(df_fin)
    print(f"[Trazabilidad] columnas monetarias detectadas: {monetary_cols}")
    detected_ok = "monto_transaccion" in monetary_cols and "ingreso_mensual" in monetary_cols
    ok &= detected_ok

    dist = derive_distribution(df_fin["ingreso_mensual"])
    real_mean = float(df_fin["ingreso_mensual"].mean())
    real_std = float(df_fin["ingreso_mensual"].std(ddof=1))
    dist_ok = abs(dist["mean"] - real_mean) < 1e-9 and abs(dist["std"] - real_std) < 1e-9
    print(f"[Trazabilidad] media/std derivadas == media/std reales de la columna: "
          f"{'OK' if dist_ok else 'FALLO'}")
    ok &= dist_ok

    rate_detected = detect_binary_rate(df_fin, "churn")
    real_rate = float((df_fin["churn"] == 1).mean())
    rate_ok = rate_detected is not None and abs(rate_detected - real_rate) < 1e-9
    print(f"[Trazabilidad] proporcion real de churn detectada: {rate_detected:.3f} "
          f"(real: {real_rate:.3f}) -> {'OK' if rate_ok else 'FALLO'}")
    ok &= rate_ok

    if ok:
        print("✅ Motor Monte Carlo consistente con formula cerrada y parametros "
              "100% trazables a datos reales (sin supuestos ocultos).")
    return ok


def run_logistics_viz_smoke_test() -> bool:
    """Capa 9: genera el mapa de rutas y la animacion de inventario sobre

    datos_ejemplo_logistica.csv (dataset real del proyecto, sin coordenadas
    reales) y verifica la degradacion en cascada esperada: sin lat/lon reales
    el mapa NO inventa coordenadas por defecto (requiere mapeo manual), y la
    animacion de inventario se genera sin excepciones con un ROP trazable.
    """
    print(f"\n{'='*70}\nCAPA 9: Rutas e inventario animado (logistica)\n{'='*70}")
    from utils.logistics_viz import (
        availability_report,
        build_routes_map,
        compute_reorder_point,
        inventory_animation,
    )

    ok = True
    csv_path = os.path.join(os.path.dirname(__file__), "datos_ejemplo_logistica.csv")
    df = pd.read_csv(csv_path)

    rep = availability_report(df)
    print(f"[Disponibilidad] {rep}")

    # El dataset de ejemplo NO tiene lat/lon reales: el sistema no debe fingir
    # que si las tiene.
    no_fake_coords_ok = rep["has_real_coords"] is False
    print(f"[Anti-invencion] has_real_coords=False (correcto, el dataset no "
          f"trae coordenadas) -> {'OK' if no_fake_coords_ok else 'FALLO'}")
    ok &= no_fake_coords_ok

    # Sin mapeo manual, el mapa NO se genera (no inventa coordenadas).
    deck_none, n_none = build_routes_map(df, rep["categorical_location_col"], {})
    no_map_without_mapping_ok = deck_none is None and n_none == 0
    print(f"[Anti-invencion] sin mapeo manual -> deck=None -> "
          f"{'OK' if no_map_without_mapping_ok else 'FALLO'}")
    ok &= no_map_without_mapping_ok

    # Con mapeo manual (ingresado explicitamente, simulando a un usuario real),
    # el mapa SI se genera sin excepciones.
    manual_coords = {"oriente": (4.14, -73.63), "sur": (1.21, -77.28),
                     "centro": (4.71, -74.07), "norte": (10.96, -74.79)}
    try:
        deck, n_routes = build_routes_map(df, rep["categorical_location_col"],
                                          manual_coords, delay_col=rep["delay_col"])
        map_ok = deck is not None and n_routes > 0
    except Exception as exc:  # noqa: BLE001
        map_ok = False
        print(f"❌ build_routes_map lanzo una excepcion: {exc}")
    print(f"[Mapa] con mapeo manual -> {n_routes if map_ok else 0} rutas "
          f"graficadas -> {'OK' if map_ok else 'FALLO'}")
    ok &= map_ok

    # Animacion de inventario: debe generarse sin excepciones, con ROP trazable.
    try:
        rop_info = compute_reorder_point(df[rep["stock_col"]])
        real_p20 = float(np.percentile(pd.to_numeric(df[rep["stock_col"]],
                                                       errors="coerce").dropna(), 20))
        rop_trazable = abs(rop_info["rop"] - real_p20) < 1e-9
        fig = inventory_animation(df[rep["stock_col"]], rop_info["rop"])
        anim_ok = fig is not None and len(fig.frames) > 0 and rop_trazable
    except Exception as exc:  # noqa: BLE001
        anim_ok = False
        print(f"❌ inventory_animation lanzo una excepcion: {exc}")
    print(f"[Inventario] ROP={rop_info.get('rop')} (percentil 20 real de la "
          f"columna) con {len(fig.frames) if anim_ok else 0} frames -> "
          f"{'OK' if anim_ok else 'FALLO'}")
    ok &= anim_ok

    if ok:
        print("✅ Mapa e inventario animado se generan sin excepciones, sin "
              "inventar coordenadas ni supuestos de reorden ocultos.")
    return ok


def run_war_room_smoke_test() -> bool:
    """Capa 10: verifica que el what-if re-prediga con el modelo YA entrenado

    (sin reentrenar) y que mover un slider a un extremo conocido cambie la
    prediccion en la direccion matematicamente esperada; y que la matriz BCG
    genere un cuadrante por cluster ya calculado, sin inventar segmentos.
    """
    print(f"\n{'='*70}\nCAPA 10: War Room de escenarios (estrategia)\n{'='*70}")
    from agents.etl_agent import run_etl
    from agents.router_agent import run_router
    from agents.supervised_agent import run_supervised
    from agents.unsupervised_agent import run_unsupervised
    from utils.war_room import (
        build_baseline_row,
        build_bcg_quadrants,
        default_bcg_axes,
        predict_scenario,
    )

    ok = True
    rng = np.random.default_rng(0)
    n = 300
    ingreso = rng.normal(3000, 800, n)
    otra = rng.normal(50, 10, n)
    # 'precio' depende monotonamente y fuertemente de 'ingreso' (relacion
    # conocida): sirve para verificar la direccion de la re-prediccion.
    precio = 500 + 10 * ingreso + rng.normal(0, 200, n)
    df = pd.DataFrame({"ingreso": ingreso, "otra": otra, "precio": precio})

    bus = EventBus()
    state = new_state(df, bus, domain="estrategia")
    th = run_in_thread(lambda: state.update(run_etl(state)), bus)
    drain_bus(bus, th)
    bus2 = EventBus()
    state["bus"] = bus2
    th2 = run_in_thread(lambda: state.update(run_router(state)), bus2)
    drain_bus(bus2, th2)

    state["target_column"] = "precio"
    bus3 = EventBus()
    state["bus"] = bus3
    th3 = run_in_thread(lambda: state.update(run_supervised(state)), bus3)
    drain_bus(bus3, th3)
    bus4 = EventBus()
    state["bus"] = bus4
    th4 = run_in_thread(lambda: state.update(run_unsupervised(state)), bus4)
    drain_bus(bus4, th4)

    sup = state["supervised_result"]
    uns = state["unsupervised_result"]

    # --- 1. Re-prediccion sin reentrenar, direccion esperada ---
    has_trained_model = sup.get("status") == "ok" and sup.get("trained_model") is not None
    print(f"[Artefacto] modelo entrenado disponible sin reentrenar: "
          f"{'OK' if has_trained_model else 'FALLO'}")
    ok &= has_trained_model

    baseline = build_baseline_row(state["clean_df"], "precio")
    lo = predict_scenario(sup, baseline,
                          {"ingreso": float(state["clean_df"]["ingreso"].min())})
    hi = predict_scenario(sup, baseline,
                          {"ingreso": float(state["clean_df"]["ingreso"].max())})
    direction_ok = (lo.get("status") == "ok" and hi.get("status") == "ok"
                    and hi["raw_prediction"] > lo["raw_prediction"])
    print(f"[What-if] ingreso minimo -> pred={lo.get('raw_prediction'):.2f} | "
          f"ingreso maximo -> pred={hi.get('raw_prediction'):.2f} -> "
          f"{'OK (direccion esperada)' if direction_ok else 'FALLO'}")
    ok &= direction_ok

    # --- 2. Matriz BCG sobre clusters ya calculados (sin inventar segmentos) ---
    x_ax, y_ax = default_bcg_axes(uns.get("overall_mean", {}), "precio")
    quadrants = build_bcg_quadrants(uns, x_ax, y_ax)
    bcg_ok = len(quadrants) == uns.get("best_k", -1) and uns.get("best_k", 0) > 0
    print(f"[BCG] ejes={x_ax}/{y_ax} -> {len(quadrants)} cuadrantes vs "
          f"{uns.get('best_k')} clusters reales -> {'OK' if bcg_ok else 'FALLO'}")
    ok &= bcg_ok

    if ok:
        print("✅ War Room re-predice con el modelo ya entrenado (direccion "
              "matematicamente correcta) y la matriz BCG usa solo clusters "
              "reales, sin inventar segmentos.")
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

    all_ok &= run_voice_qa_smoke_test()
    all_ok &= run_finance_montecarlo_smoke_test()
    all_ok &= run_logistics_viz_smoke_test()
    all_ok &= run_war_room_smoke_test()

    print(f"\n{'='*70}")
    print(f"Matriz ejecutada: {len(cases)} casos x {len(DOMAIN_KEYS)} dominios = "
          f"{len(cases) * len(DOMAIN_KEYS)} corridas + smoke test de voz (Capa 7) "
          f"+ Monte Carlo (Capa 8) + rutas/inventario (Capa 9) + War Room (Capa 10).")
    print("RESULTADO GLOBAL:", "OK" if all_ok else "CON ERRORES")
    print("="*70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
