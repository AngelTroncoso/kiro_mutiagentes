"""Analista de Datos Multiagente - interfaz Streamlit con streaming pedagogico.

Flujo de la app:
    0. El usuario elige un dominio de negocio (Finanzas / Logistica / Estrategia).
       Esto SOLO reconfigura vocabulario y sugerencias; nunca la matematica.
    1. El usuario sube un CSV/XLSX.
    2. "Iniciar diagnostico": corren ETL + Router narrando en tiempo real.
    3. El usuario ajusta objetivo, enfoques a ejecutar y objetivo de negocio.
    4. "Ejecutar analisis": corren los agentes elegidos + recomendador, en streaming.
    5. Se muestran resultados en un panel reordenable y una recomendacion destacada.
    6. Reporte ejecutivo gerencial, descargable.

La narracion llega desde un hilo de fondo (los agentes) a traves de un EventBus,
y se consume aqui como un generador de eventos que se renderiza en vivo.
"""

from __future__ import annotations

import itertools
import os
import sys

# Aseguramos que el paquete sea importable al ejecutar `streamlit run app.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from agents.recommender_agent import GOAL_LABELS
from agents.voice_qa_agent import answer_question
from orchestrator.graph import run_analysis_phase, run_diagnosis_phase
from orchestrator.state import new_state
from utils.data_loading import basic_overview, load_dataframe
from utils.domain_config import DOMAIN_KEYS, get_domain, goal_labels_for
from utils.llm import is_llm_available
from utils.streaming import EventBus, run_in_thread
from utils.finance_sim import (
    MonteCarloInputs,
    derive_distribution,
    derive_split_distribution,
    detect_binary_rate,
    detect_monetary_columns,
    run_monte_carlo,
)
from utils.logistics_viz import (
    availability_report as logistics_availability_report,
    build_routes_map,
    compute_reorder_point,
    inventory_animation,
)
from utils.theme import apply_theme, render_domain_badge, render_stepper
from utils.visualization import bcg_matrix_scatter, cashflow_fan_chart, npv_distribution_hist
from utils.voice import synthesize_speech, transcribe_audio, voice_status
from utils.war_room import (
    build_baseline_row,
    build_bcg_quadrants,
    default_bcg_axes,
    predict_scenario,
    summarize_bcg_with_llm,
)

st.set_page_config(page_title="Analista de Datos Multiagente",
                   page_icon="🤖", layout="wide")

_key_counter = itertools.count()

PHASES = ["Dominio", "Carga", "Diagnostico", "Analisis", "Recomendacion", "Reporte"]

# --- Grilla reordenable de la Capa 3 (opcional, con degradacion segura) -------
try:
    from streamlit_sortables import sort_items

    _SORTABLES_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SORTABLES_AVAILABLE = False

# --- Captura de audio para la Capa 7 (con degradacion en 2 niveles) ---------
# Nivel 1 (preferido): st.audio_input, nativo de Streamlit >= 1.31. No depende
# de un componente de terceros, asi que el navegador SI muestra el prompt de
# permiso de microfono de forma confiable.
_NATIVE_AUDIO_INPUT = hasattr(st, "audio_input")

# Nivel 2 (respaldo): audio-recorder-streamlit, un componente de terceros que
# en algunos navegadores/versiones de Streamlit no solicita el permiso de
# microfono correctamente (iframe sin `allow="microphone"`). Se usa solo si
# el nativo no existe (Streamlit viejo).
try:
    from audio_recorder_streamlit import audio_recorder

    _AUDIO_RECORDER_AVAILABLE = True
except Exception:  # noqa: BLE001
    _AUDIO_RECORDER_AVAILABLE = False

_AUDIO_INPUT_AVAILABLE = _NATIVE_AUDIO_INPUT or _AUDIO_RECORDER_AVAILABLE


def _next_key(prefix: str) -> str:
    return f"{prefix}_{next(_key_counter)}"


def _current_stage_index() -> int:
    stage = st.session_state.get("stage", "domain")
    return {
        "domain": 0, "upload": 1, "loaded": 1, "diagnosed": 2,
        "analyzed": 3,
    }.get(stage, 0)


# ----------------------------------------------------------------------
# Consumo del stream de eventos y renderizado en vivo
# ----------------------------------------------------------------------
def consume_stream(bus: EventBus, thread) -> dict:
    """Renderiza los eventos en vivo y devuelve los artefactos coleccionados.

    Artefactos por agente: {agent: {"charts": [...], "tables": [...],
    "results": {...}, "narration": [str], "warnings": [str], "summary": str}}
    """
    collected: dict[str, dict] = {}

    def bucket(agent: str) -> dict:
        return collected.setdefault(
            agent,
            {"charts": [], "tables": [], "results": {}, "narration": [],
             "warnings": [], "summary": ""},
        )

    for event in bus.stream(is_alive=thread.is_alive):
        etype = event.get("type")
        agent = event.get("agent", "")

        if etype == "agent_start":
            st.markdown(f"### {event.get('icon','*')} {event.get('title', agent)}")

        elif etype == "narration":
            text = st.write_stream(event["stream"])
            if text:
                bucket(agent)["narration"].append(text)

        elif etype == "text":
            style = event.get("style")
            if style == "caption":
                st.caption(event["text"])
            else:
                st.markdown(event["text"])

        elif etype == "warning":
            st.warning(event["text"])
            bucket(agent)["warnings"].append(event["text"])

        elif etype == "chart":
            fig = event["figure"]
            st.plotly_chart(fig, width="stretch", key=_next_key("live_chart"))
            if event.get("caption"):
                st.caption(event["caption"])
            bucket(agent)["charts"].append((event.get("title", ""), fig,
                                            event.get("caption", "")))

        elif etype == "table":
            st.markdown(f"**{event.get('title','')}**")
            st.dataframe(event["data"], width="stretch", hide_index=True)
            bucket(agent)["tables"].append((event.get("title", ""), event["data"]))

        elif etype == "result":
            bucket(agent)["results"][event.get("key", "result")] = event.get("payload", {})

        elif etype == "agent_end":
            summary = event.get("summary", "")
            bucket(agent)["summary"] = summary
            if summary:
                st.success(summary)
            st.divider()

        elif etype == "error":
            st.error(event.get("text", "Error en el pipeline."))
            with st.expander("Detalle tecnico"):
                st.code(event.get("traceback", ""))

    thread.join(timeout=1.0)
    return collected


def report_to_markdown(report: dict) -> str:
    """Serializa el reporte ejecutivo a Markdown para descargar/compartir."""
    lines: list[str] = []
    lines.append(f"# {report.get('titulo', 'Reporte Ejecutivo')}")
    lines.append("")
    meta = report.get("_meta", {})
    if meta:
        lines.append(f"_Dominio: {meta.get('domain_label', 'General')}. "
                     f"Universo analizado: {meta.get('n_rows', '?')} registros, "
                     f"{meta.get('n_cols', '?')} variables. Variable de interes: "
                     f"{meta.get('target', 'n/d')}._")
        lines.append("")
    lines.append("## Resumen de la situacion")
    lines.append(report.get("resumen_situacion", ""))
    lines.append("")
    lines.append("## Hallazgo clave")
    lines.append(f"> {report.get('hallazgo_clave', '')}")
    lines.append("")
    lines.append("## Recomendaciones")
    for i, rec in enumerate(report.get("recomendaciones", []), start=1):
        lines.append(f"### {i}. {rec.get('titulo', '')}")
        lines.append(rec.get("descripcion", ""))
        lines.append("")
        lines.append(f"- **Impacto esperado:** {rec.get('impacto_esperado', '')}")
        lines.append(f"- **Consideracion / riesgo:** {rec.get('riesgo_o_consideracion', '')}")
        lines.append("")
    lines.append("## Siguiente paso sugerido")
    lines.append(report.get("siguiente_paso_sugerido", ""))
    lines.append("")
    lines.append("---")
    lines.append("_Generado por Analista de Datos Multiagente. Todas las cifras provienen "
                 "del dataset analizado y de las metricas calculadas por el sistema._")
    return "\n".join(lines)


def _render_collected(collected: dict) -> None:
    """Renderiza narracion + graficos + tablas ya coleccionados (post-run)."""
    for agent, data in collected.items():
        if not any([data["narration"], data["charts"], data["tables"], data["warnings"]]):
            continue
        with st.expander(f"Detalle - {agent}", expanded=False):
            for text in data["narration"]:
                st.markdown(text)
            for w in data["warnings"]:
                st.warning(w)
            for title, fig, cap in data["charts"]:
                st.plotly_chart(fig, width="stretch", key=_next_key("re_chart"))
                if cap:
                    st.caption(cap)
            for title, tbl in data["tables"]:
                st.markdown(f"**{title}**")
                st.dataframe(tbl, width="stretch", hide_index=True)


def _render_approach_panel(agent_name: str, result: dict | None, collected: dict) -> None:
    """Contenido de una tarjeta de resultado (Supervisado/No supervisado/Refuerzo)."""
    if not result or result.get("status") != "ok":
        reason = (result or {}).get("reason", "No se ejecuto este analisis.")
        st.info(f"Sin resultados. {reason}")
        return
    data = collected.get(agent_name, {})
    for text in data.get("narration", []):
        st.markdown(text)
    for title, fig, cap in data.get("charts", []):
        st.plotly_chart(fig, width="stretch", key=_next_key("panel_chart"))
        if cap:
            st.caption(cap)
    for title, tbl in data.get("tables", []):
        st.markdown(f"**{title}**")
        st.dataframe(tbl, width="stretch", hide_index=True)


# ----------------------------------------------------------------------
# Estado inicial + Capa 0: selector de dominio
# ----------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "domain"
if "domain" not in st.session_state:
    st.session_state.domain = None

active_domain = st.session_state.get("domain")
domain_cfg = apply_theme(active_domain)

st.title("🤖 Analista de Datos Multiagente")
st.caption("Sube un dataset y deja que un equipo de agentes lo limpie, lo analice y "
           "te explique cada paso en el lenguaje de tu negocio.")

render_stepper(PHASES, _current_stage_index())

with st.sidebar:
    st.header("Estado del sistema")
    if is_llm_available():
        st.success("Groq conectado: narracion con IA en streaming.")
    else:
        st.info("Sin GROQ_API_KEY: uso un narrador local (igual de pedagogico).\n\n"
                "Para activar la IA, define la variable de entorno `GROQ_API_KEY`.")
    st.divider()
    if active_domain:
        st.markdown(f"**Dominio activo:** {domain_cfg['icon']} {domain_cfg['label']}")
        if st.button("Cambiar de dominio", width="stretch"):
            st.session_state.domain = None
            st.session_state.stage = "domain"
            st.rerun()
    st.divider()
    st.markdown("**Las 6 capas**")
    st.markdown(
        "- 🧹 **ETL**: limpieza narrada\n"
        "- 🧭 **Router**: diagnostico\n"
        "- 🎯 **Supervisado**: predecir\n"
        "- 🔍 **No supervisado**: agrupar\n"
        "- 🕹️ **Refuerzo**: decidir (bandit)\n"
        "- 🏆 **Recomendador**: veredicto final\n"
        "- 📄 **Reporte ejecutivo**: cierre gerencial"
    )
    if st.button("Reiniciar todo", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ----------------------------------------------------------------------
# Paso 0: elegir dominio de negocio (bloquea el resto hasta elegir)
# ----------------------------------------------------------------------
if not active_domain:
    st.subheader("0. Elige el dominio de negocio")
    st.caption("Esto solo cambia el vocabulario, las sugerencias y el reporte final. "
               "Las metricas y los modelos son exactamente los mismos en cualquier dominio.")

    cols = st.columns(3)
    for col, key in zip(cols, DOMAIN_KEYS):
        cfg = get_domain(key)
        with col:
            with st.container(border=True):
                st.markdown(f"### {cfg['icon']} {cfg['label']}")
                st.caption(cfg["tagline"])
                if st.button(f"Elegir {cfg['label']}", key=f"pick_{key}",
                            width="stretch"):
                    st.session_state.domain = key
                    st.session_state.stage = "upload"
                    st.rerun()
    st.stop()

render_domain_badge(domain_cfg)

# ----------------------------------------------------------------------
# Paso 1: carga de archivo
# ----------------------------------------------------------------------
st.subheader("1. Carga tu dataset")
sample_name = domain_cfg.get("sample_dataset")
if sample_name:
    st.caption(f"Sugerencia: prueba con `{sample_name}` (incluido en el proyecto) "
               f"para ver el dominio {domain_cfg['label']} en accion.")

uploaded = st.file_uploader("Formatos aceptados: CSV o Excel (.xlsx)",
                            type=["csv", "xlsx", "xls"])

if uploaded is not None:
    file_sig = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("file_sig") != file_sig:
        try:
            df = load_dataframe(uploaded.getvalue(), uploaded.name)
            st.session_state.raw_df = df
            st.session_state.filename = uploaded.name
            st.session_state.file_sig = file_sig
            st.session_state.stage = "loaded"
            # Limpiamos resultados previos.
            for k in ("diagnosis", "clean_df", "collected_diag",
                      "analysis_results", "collected_analysis", "recommendation",
                      "executive_report", "panel_order"):
                st.session_state.pop(k, None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No pude leer el archivo: {exc}")

if "raw_df" in st.session_state:
    df = st.session_state.raw_df
    ov = basic_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", ov["n_rows"])
    c2.metric("Columnas", ov["n_cols"])
    c3.metric("Celdas vacias", ov["total_missing"])
    c4.metric("Duplicados", ov["n_duplicates"])
    with st.expander("Vista previa de los datos crudos", expanded=False):
        st.dataframe(df.head(20), width="stretch")

    # ------------------------------------------------------------------
    # Paso 2: diagnostico (ETL + Router)
    # ------------------------------------------------------------------
    st.subheader("2. Diagnostico y limpieza")
    if st.session_state.stage == "loaded":
        if st.button("▶ Iniciar diagnostico", type="primary"):
            bus = EventBus()
            state = new_state(df.copy(), bus, domain=active_domain)

            def _job():
                result = run_diagnosis_phase(state)
                # Guardamos lo relevante en el propio state (el hilo lo comparte).
                state.update(result)

            thread = run_in_thread(_job, bus)
            with st.container(border=True):
                collected = consume_stream(bus, thread)

            st.session_state.clean_df = state.get("clean_df")
            st.session_state.diagnosis = state.get("diagnosis", {})
            st.session_state.etl_report = state.get("etl_report", {})
            st.session_state.collected_diag = collected
            st.session_state.stage = "diagnosed"
            st.rerun()
    else:
        st.success("Diagnostico completado. Revisa los graficos de limpieza mas abajo "
                   "o vuelve a ejecutarlo desde 'Reiniciar todo'.")
        # Re-render de artefactos del diagnostico.
        collected = st.session_state.get("collected_diag", {})
        _render_collected(collected)


# ----------------------------------------------------------------------
# Paso 3 y 4: seleccion de analisis + ejecucion
# ----------------------------------------------------------------------
if st.session_state.get("stage") in ("diagnosed", "analyzed"):
    diagnosis = st.session_state.get("diagnosis", {})
    clean_df: pd.DataFrame = st.session_state.get("clean_df")
    applicable = diagnosis.get("applicable", [])

    if diagnosis.get("domain_match") is False:
        st.warning(f"⚠️ Este dataset no calza claramente con columnas tipicas de "
                  f"{domain_cfg['label']}. Las sugerencias de dominio pueden no "
                  f"corresponder; revisa con cautela.")

    st.subheader("3. Elige que analisis ejecutar")
    st.caption("Deje pre-seleccionados los enfoques recomendados por el diagnostico. "
               "Puedes cambiarlos libremente.")

    label_map = {
        "supervised": "🎯 Supervisado (predecir un valor/categoria)",
        "unsupervised": "🔍 No supervisado (descubrir grupos)",
        "reinforcement": "🕹️ Refuerzo (aprender a decidir - bandit)",
    }
    target_frame = diagnosis.get("target_frame")

    with st.form("analysis_form"):
        cols = st.columns(3)
        chosen: list[str] = []
        for i, key in enumerate(["supervised", "unsupervised", "reinforcement"]):
            info = diagnosis.get(key, {})
            default = key in applicable
            with cols[i]:
                pick = st.checkbox(label_map[key], value=default,
                                   help=info.get("reason", ""))
                if pick:
                    chosen.append(key)
                st.caption(info.get("reason", ""))

        c1, c2 = st.columns(2)
        with c1:
            all_cols = list(clean_df.columns) if clean_df is not None else []
            default_target = diagnosis.get("target_column")
            t_index = all_cols.index(default_target) if default_target in all_cols else 0
            target_label = "Variable objetivo (para supervisado/refuerzo)"
            if target_frame:
                target_label += f" · enmarcada como: {target_frame}"
            target = st.selectbox(target_label,
                                  options=all_cols or ["(sin columnas)"],
                                  index=t_index if all_cols else 0)
        with c2:
            goal_labels = goal_labels_for(active_domain)
            goal_keys = list(goal_labels.keys())
            goal = st.selectbox(
                "Tu objetivo de negocio (guia la recomendacion final)",
                options=goal_keys,
                format_func=lambda k: goal_labels.get(k, GOAL_LABELS.get(k, k)),
                index=goal_keys.index("predecir") if "predecir" in goal_keys else 0,
            )

        submitted = st.form_submit_button("▶ Ejecutar analisis", type="primary")

    if submitted:
        if not chosen:
            st.warning("Selecciona al menos un tipo de analisis.")
        else:
            bus = EventBus()
            state = new_state(clean_df.copy(), bus, domain=active_domain)
            state["clean_df"] = clean_df.copy()
            state["diagnosis"] = diagnosis
            state["selected_analyses"] = chosen
            state["target_column"] = target
            state["business_goal"] = goal

            def _job2():
                result = run_analysis_phase(state)
                state.update(result)

            thread = run_in_thread(_job2, bus)
            st.markdown("### Ejecucion en vivo")
            with st.container(border=True):
                collected = consume_stream(bus, thread)

            st.session_state.analysis_results = {
                "supervised": state.get("supervised_result"),
                "unsupervised": state.get("unsupervised_result"),
                "reinforcement": state.get("reinforcement_result"),
            }
            st.session_state.recommendation = state.get("recommendation")
            st.session_state.executive_report = state.get("executive_report")
            st.session_state.collected_analysis = collected
            st.session_state.stage = "analyzed"
            st.session_state.pop("panel_order", None)
            st.rerun()


# ----------------------------------------------------------------------
# Paso 5: resultados estructurados (panel reordenable) + recomendacion
# ----------------------------------------------------------------------
if st.session_state.get("stage") == "analyzed":
    st.subheader("4. Resultados por enfoque")
    collected = st.session_state.get("collected_analysis", {})
    results = st.session_state.get("analysis_results", {})

    panel_specs = {
        "supervised": ("🎯 Supervisado", "Supervisado"),
        "unsupervised": ("🔍 No supervisado", "NoSupervisado"),
        "reinforcement": ("🕹️ Refuerzo", "Refuerzo"),
    }
    available_panels = [k for k in panel_specs if results.get(k) is not None]

    if _SORTABLES_AVAILABLE and available_panels:
        st.caption("↕️ Arrastra las tarjetas para reordenarlas segun lo que te "
                  "importe revisar primero.")
        try:
            default_order = st.session_state.get("panel_order") or [
                panel_specs[k][0] for k in available_panels
            ]
            new_order = sort_items(default_order, key="panel_sortable")
            st.session_state.panel_order = new_order
            label_to_key = {v[0]: k for k, v in panel_specs.items()}
            ordered_keys = [label_to_key[lbl] for lbl in new_order
                            if lbl in label_to_key]
        except Exception as exc:  # noqa: BLE001
            # Degradacion segura: si el componente falla en runtime, usamos el
            # orden estatico original sin interrumpir la app.
            st.caption(f"(Panel arrastrable no disponible: {exc}. Orden fijo.)")
            ordered_keys = available_panels
    else:
        if not _SORTABLES_AVAILABLE:
            st.caption("(streamlit-sortables no disponible: mostrando orden fijo.)")
        ordered_keys = available_panels

    panel_cols = st.columns(len(ordered_keys)) if ordered_keys else []
    for col, key in zip(panel_cols, ordered_keys):
        title, agent_name = panel_specs[key]
        with col:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                _render_approach_panel(agent_name, results.get(key), collected)

    # -------------------- Recomendacion destacada --------------------
    st.subheader("5. Recomendacion final")
    rec = st.session_state.get("recommendation")
    if rec and rec.get("status") == "ok":
        st.success(f"### 🏆 {rec['winner_label']}\n\n"
                   f"Para tu objetivo **\"{rec['business_goal_label']}\"**, "
                   f"el enfoque recomendado es **{rec['winner_label']}** "
                   f"({rec.get('winner_detail','')}).")
        rec_data = st.session_state.get("collected_analysis", {}).get("Recomendador", {})
        for text in rec_data.get("narration", []):
            st.markdown(text)
        for title, fig, cap in rec_data.get("charts", []):
            st.plotly_chart(fig, width="stretch", key=_next_key("rec_chart"))
            if cap:
                st.caption(cap)
        with st.expander("Como se calculo la recomendacion"):
            st.markdown(rec.get("ranking_text", ""))
            score_df = pd.DataFrame({
                "Enfoque": list(rec["raw_scores"].keys()),
                "Calidad (0-1)": [round(v, 3) for v in rec["raw_scores"].values()],
                "Peso objetivo": [rec["weights"][k] for k in rec["raw_scores"]],
                "Score final": [rec["final_scores"][k] for k in rec["raw_scores"]],
            })
            st.dataframe(score_df, width="stretch", hide_index=True)
    else:
        st.info("Aun no hay recomendacion disponible.")

    # -------------------- Reporte Ejecutivo Gerencial (Capa 6) --------------------
    st.subheader("6. Reporte ejecutivo")
    st.caption(f"Traduccion de los hallazgos a decisiones de negocio para "
               f"{domain_cfg['label']}. Tono ejecutivo, sin tecnicismos.")
    report = st.session_state.get("executive_report")
    if report and report.get("recomendaciones"):
        with st.container(border=True):
            st.markdown(f"## 📄 {report.get('titulo', 'Reporte Ejecutivo')}")
            meta = report.get("_meta", {})
            if meta:
                st.caption(f"Dominio: {meta.get('domain_label','General')} · "
                           f"Universo: {meta.get('n_rows','?')} registros · "
                           f"{meta.get('n_cols','?')} variables · "
                           f"Variable de interes: {meta.get('target','n/d')} · "
                           f"Enfoque: {meta.get('winner_label','n/d')}")

            st.markdown("**Resumen de la situacion**")
            st.write(report.get("resumen_situacion", ""))

            st.info(f"**Hallazgo clave:** {report.get('hallazgo_clave', '')}")

            st.markdown("**Recomendaciones para la gerencia**")
            rec_cols = st.columns(3)
            for col, rec in zip(rec_cols, report.get("recomendaciones", [])):
                with col:
                    with st.container(border=True):
                        st.markdown(f"##### {rec.get('titulo', '')}")
                        st.write(rec.get("descripcion", ""))
                        st.markdown(f"**📈 Impacto esperado**\n\n{rec.get('impacto_esperado', '')}")
                        st.markdown(f"**⚠️ Consideracion**\n\n{rec.get('riesgo_o_consideracion', '')}")

            st.success(f"**Siguiente paso sugerido:** "
                       f"{report.get('siguiente_paso_sugerido', '')}")

            md = report_to_markdown(report)
            fname = f"reporte_ejecutivo_{(meta.get('target') or 'analisis')}.md"
            st.download_button(
                "⬇️ Descargar reporte (Markdown)",
                data=md.encode("utf-8"),
                file_name=fname,
                mime="text/markdown",
                width="stretch",
            )
    else:
        st.info("El reporte ejecutivo se genera automaticamente al finalizar el analisis.")

    # ============================================================
    # Innovacion - Capa 7: Chat de voz con el Analista
    # ============================================================
    st.divider()
    st.subheader("🎙️ Innovacion — Preguntale al Analista")
    st.caption("Preguntas en lenguaje natural sobre lo que YA se calculo. Las "
              "respuestas citan siempre el agente fuente y nunca inventan cifras.")

    vstatus = voice_status()
    status_bits = []
    status_bits.append("🎤 Voz->texto (Groq Whisper): " +
                       ("activo" if vstatus["stt"] else "no disponible (usa texto)"))
    status_bits.append("🔊 Texto->voz: " +
                       ("ElevenLabs" if vstatus["tts_elevenlabs"] else "gTTS (respaldo)"))
    st.caption(" · ".join(status_bits))

    voice_state = {
        "etl_report": st.session_state.get("etl_report", {}),
        "diagnosis": st.session_state.get("diagnosis", {}),
        "supervised_result": results.get("supervised"),
        "unsupervised_result": results.get("unsupervised"),
        "reinforcement_result": results.get("reinforcement"),
        "recommendation": st.session_state.get("recommendation"),
        "executive_report": st.session_state.get("executive_report"),
    }

    hands_free = st.toggle("🖐️ Modo manos libres (transcribe y responde en audio "
                          "automaticamente)", value=False,
                          help="Accesibilidad: transcripcion en vivo + respuesta "
                               "en audio y texto simultaneo.")

    with st.container(border=True):
        question_text = None

        if _NATIVE_AUDIO_INPUT:
            st.caption("🎤 Grava tu pregunta con el microfono (el navegador te "
                      "pedira permiso la primera vez):")
            audio_value = st.audio_input("Pregunta por voz",
                                         key=_next_key("voice_recorder"),
                                         label_visibility="collapsed")
            audio_bytes = audio_value.getvalue() if audio_value is not None else None
            if audio_bytes:
                transcribed = transcribe_audio(audio_bytes)
                if transcribed:
                    st.success(f"🎤 Transcrito: \"{transcribed}\"")
                    question_text = transcribed
                else:
                    st.warning("No pude transcribir el audio (sin GROQ_API_KEY o "
                              "fallo de red). Escribe tu pregunta abajo.")
        elif _AUDIO_RECORDER_AVAILABLE:
            st.caption("🎤 Grava tu pregunta (haz clic en el icono; si el "
                      "navegador no pide permiso de microfono, usa el campo de "
                      "texto de abajo):")
            audio_bytes = audio_recorder(text="", icon_size="2x",
                                         key=_next_key("voice_recorder"))
            if audio_bytes:
                transcribed = transcribe_audio(audio_bytes)
                if transcribed:
                    st.success(f"🎤 Transcrito: \"{transcribed}\"")
                    question_text = transcribed
                else:
                    st.warning("No pude transcribir el audio (sin GROQ_API_KEY o "
                              "fallo de red). Escribe tu pregunta abajo.")
        else:
            st.caption("(Captura de audio no disponible en este entorno: usa el "
                      "campo de texto.)")

        typed = st.text_input(
            "O escribe tu pregunta",
            placeholder="Ej: ¿cual fue el mejor modelo? / ¿que variable importa mas?",
            key=_next_key("voice_text_input"),
        )
        ask_clicked = st.button("Preguntar", key=_next_key("voice_ask_btn"))

        final_question = question_text or (typed if ask_clicked and typed else None)
        if question_text or (ask_clicked and typed):
            final_question = question_text or typed
            with st.spinner("Consultando el estado del analisis..."):
                qa_result = answer_question(final_question, voice_state)

            st.markdown(f"**🗣️ Pregunta:** {final_question}")
            st.info(f"**🤖 Respuesta:** {qa_result['answer']}")
            st.caption(f"Fuente: {qa_result['source_agent']} · "
                      f"motor: {qa_result['engine']}")

            if hands_free:
                audio_out, engine_out = synthesize_speech(qa_result["answer"])
                if audio_out:
                    st.audio(audio_out, format="audio/mp3")
                    st.caption(f"🔊 Audio generado con {engine_out}.")
                else:
                    st.caption("(Sin motor de voz disponible: solo texto.)")

    # ============================================================
    # Innovacion - Capa 8: Simulador Monte Carlo de flujo de caja
    # Solo visible en dominio Finanzas (o General, con aviso).
    # ============================================================
    if active_domain in ("finanzas", "general"):
        st.divider()
        st.subheader("💰 Innovacion — Proyeccion de flujo de caja (Monte Carlo)")
        if active_domain == "general":
            st.warning("Este simulador esta pensado para el dominio Finanzas. "
                      "Puedes usarlo igual, pero interpreta los resultados con "
                      "cautela fuera de ese contexto.")
        st.caption("Cada parametro de la simulacion sale de una columna real del "
                  "dataset (media y desviacion estandar reales) o de un valor que "
                  "tu fijas explicitamente. Ningun supuesto queda oculto.")

        mc_df: pd.DataFrame = st.session_state.get("clean_df")
        if mc_df is None or mc_df.empty:
            st.info("Carga y limpia un dataset primero para habilitar la simulacion.")
        else:
            numeric_cols = mc_df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.info("El dataset no tiene columnas numericas: no se puede "
                       "derivar una distribucion de flujo de caja.")
            else:
                suggested = detect_monetary_columns(mc_df)
                default_col = suggested[0] if suggested else numeric_cols[0]

                with st.form("montecarlo_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        cash_col = st.selectbox(
                            "Columna del dataset que representa el flujo de caja "
                            "por periodo" + (" (sugerida por nombre)" if suggested else ""),
                            options=numeric_cols,
                            index=numeric_cols.index(default_col),
                        )
                        initial_capital = st.number_input(
                            "Capital inicial", min_value=0.0, value=100_000.0, step=1000.0)
                        horizon = st.slider("Horizonte (numero de periodos)",
                                           min_value=3, max_value=60, value=12)
                    with c2:
                        discount_rate_pct = st.slider(
                            "Tasa de descuento por periodo (%)",
                            min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                        n_iterations = st.select_slider(
                            "Iteraciones Monte Carlo",
                            options=[5000, 8000, 10000], value=8000)
                        apply_churn = False
                        churn_rate_preview = None
                        target_col_mc = st.session_state.get("diagnosis", {}).get("target_column")
                        if target_col_mc:
                            churn_rate_preview = detect_binary_rate(mc_df, target_col_mc)
                        if churn_rate_preview is not None:
                            apply_churn = st.checkbox(
                                f"Ajustar entradas con la tasa real de "
                                f"'{target_col_mc}' detectada ({churn_rate_preview*100:.1f}%), "
                                f"proveniente del Agente Supervisado/Router",
                                value=(st.session_state.get("recommendation", {})
                                      .get("winner") == "supervised"),
                            )

                    submitted_mc = st.form_submit_button("▶ Correr simulacion",
                                                          type="primary")

                if submitted_mc:
                    split = derive_split_distribution(mc_df[cash_col])
                    inflow, outflow = split["inflow"], split["outflow"]
                    source_note = (
                        f"Columna '{cash_col}': {split['mode']}, "
                        f"n={inflow['n']} entradas (media={inflow['mean']:.2f}, "
                        f"std={inflow['std']:.2f})"
                        + (f", n={outflow['n']} salidas (media={outflow['mean']:.2f}, "
                           f"std={outflow['std']:.2f})" if outflow["n"] else "")
                    )
                    mc_inputs = MonteCarloInputs(
                        initial_capital=initial_capital,
                        horizon_periods=horizon,
                        discount_rate=discount_rate_pct / 100.0,
                        inflow_mean=inflow["mean"], inflow_std=inflow["std"],
                        outflow_mean=outflow["mean"], outflow_std=outflow["std"],
                        n_iterations=n_iterations,
                        churn_adjustment_rate=(churn_rate_preview if apply_churn else None),
                        source_note=source_note,
                    )
                    mc_result = run_monte_carlo(mc_inputs)
                    st.session_state.montecarlo_result = mc_result

                mc_result = st.session_state.get("montecarlo_result")
                if mc_result is not None:
                    st.caption(f"Parametros trazables: {mc_result.inputs.source_note}")
                    if mc_result.inputs.churn_adjustment_rate:
                        st.caption(
                            f"Ajuste aplicado: entradas reducidas en "
                            f"{mc_result.inputs.churn_adjustment_rate*100:.1f}% "
                            f"(tasa real detectada por el Router/Supervisado).")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("VAN mediana (P50)", f"{mc_result.npv_p50:,.0f}")
                    m2.metric("VAN P10 / P90",
                             f"{mc_result.npv_p10:,.0f} / {mc_result.npv_p90:,.0f}")
                    irr_txt = (f"{mc_result.irr_representative*100:.1f}%"
                              if mc_result.irr_representative is not None else "n/d")
                    m3.metric("TIR (sobre flujo promedio)", irr_txt)
                    m4.metric("Prob. de flujo negativo",
                             f"{mc_result.prob_flujo_negativo*100:.1f}%")

                    st.plotly_chart(
                        cashflow_fan_chart(mc_result.paths, mc_result.inputs.initial_capital),
                        width="stretch", key=_next_key("mc_fan_chart"))
                    st.caption("Banda P10-P90 del flujo de caja acumulado a lo "
                              "largo del horizonte, con una muestra de trayectorias "
                              "individuales de fondo.")

                    st.plotly_chart(
                        npv_distribution_hist(mc_result.npv_samples),
                        width="stretch", key=_next_key("mc_npv_hist"))
                    st.caption("Distribucion completa del VAN sobre todas las "
                              "iteraciones simuladas.")

    # ============================================================
    # Innovacion - Capa 9: Rutas e inventario animado
    # Solo visible en dominio Logistica (o General, con aviso).
    # ============================================================
    if active_domain in ("logistica", "general"):
        st.divider()
        st.subheader("🚚 Innovacion — Rutas e inventario animado")
        if active_domain == "general":
            st.warning("Este modulo esta pensado para el dominio Logistica. "
                      "Puedes usarlo igual, pero interpreta los resultados con "
                      "cautela fuera de ese contexto.")

        log_df: pd.DataFrame = st.session_state.get("clean_df")
        if log_df is None or log_df.empty:
            st.info("Carga y limpia un dataset primero para habilitar este modulo.")
        else:
            log_rep = logistics_availability_report(log_df)

            # --- Mapa de rutas ---
            st.markdown("**Mapa de rutas**")
            if log_rep["has_real_coords"]:
                st.caption(f"Coordenadas reales detectadas: "
                          f"'{log_rep['lat_col']}' / '{log_rep['lon_col']}'.")
                try:
                    import pydeck as pdk

                    map_df = log_df[[log_rep["lat_col"], log_rep["lon_col"]]].dropna()
                    st.pydeck_chart(pdk.Deck(
                        layers=[pdk.Layer("ScatterplotLayer", data=map_df,
                                          get_position=f"[{log_rep['lon_col']}, "
                                                       f"{log_rep['lat_col']}]",
                                          get_radius=200, get_fill_color=[139, 92, 246])],
                        initial_view_state=pdk.ViewState(
                            latitude=float(map_df[log_rep["lat_col"]].mean()),
                            longitude=float(map_df[log_rep["lon_col"]].mean()),
                            zoom=4),
                    ))
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"(No se pudo renderizar el mapa: {exc})")
            elif log_rep["categorical_location_col"]:
                loc_col = log_rep["categorical_location_col"]
                st.caption(
                    f"El dataset no tiene columnas de latitud/longitud, pero "
                    f"tiene '{loc_col}' (categorica). Asigna coordenadas para "
                    f"cada valor para visualizar el mapa; sin este mapeo, no se "
                    f"inventa ninguna coordenada.")
                unique_vals = sorted(str(v) for v in log_df[loc_col].dropna().unique())[:12]
                with st.form("logistics_map_form"):
                    manual_coords: dict[str, tuple[float, float]] = {}
                    for val in unique_vals:
                        c1, c2 = st.columns(2)
                        lat_v = c1.number_input(f"Latitud de '{val}'", value=0.0,
                                                format="%.4f", key=_next_key(f"lat_{val}"))
                        lon_v = c2.number_input(f"Longitud de '{val}'", value=0.0,
                                                format="%.4f", key=_next_key(f"lon_{val}"))
                        manual_coords[val] = (lat_v, lon_v)
                    map_submit = st.form_submit_button("▶ Generar mapa de rutas")
                if map_submit:
                    valid_coords = {k: v for k, v in manual_coords.items()
                                    if v != (0.0, 0.0)}
                    deck, n_routes = build_routes_map(
                        log_df, loc_col, valid_coords,
                        delay_col=log_rep.get("delay_col"))
                    if deck is not None:
                        st.pydeck_chart(deck)
                        st.caption(f"{n_routes} rutas graficadas. Color coral = "
                                  f"con retraso; cian = a tiempo (segun "
                                  f"'{log_rep.get('delay_col')}').")
                    else:
                        st.warning("Ingresa al menos una coordenada distinta de "
                                  "(0, 0) para generar el mapa.")
            else:
                st.info("No hay columnas de ubicacion (ni coordenadas ni "
                       "categoricas tipo ciudad/ruta) en este dataset: el mapa "
                       "queda oculto en vez de mostrar datos inventados.")

            # --- Animacion de inventario ---
            st.markdown("**Evolucion de inventario**")
            if log_rep["inventory_possible"]:
                stock_col = log_rep["stock_col"]
                rop_info = compute_reorder_point(log_df[stock_col])
                st.caption(
                    f"Punto de reorden calculado como el percentil "
                    f"{rop_info['percentile']:.0f} REAL de '{stock_col}' "
                    f"({rop_info['rop']:,.0f}), entre un minimo de "
                    f"{rop_info['min']:,.0f} y un maximo de {rop_info['max']:,.0f} "
                    f"observados. No es una formula de demanda x lead time "
                    f"(el dataset no tiene consumo diario); es un metodo simple "
                    f"y transparente sobre los datos disponibles.")
                fig_inv = inventory_animation(log_df[stock_col], rop_info["rop"])
                st.plotly_chart(fig_inv, width="stretch", key=_next_key("inv_anim"))
                st.caption("Eje X: orden de los registros en el dataset limpio "
                          "(no hay columna de fecha real). Puntos en color de "
                          "alerta = stock por debajo del punto de reorden.")
            else:
                st.info("No hay una columna de stock/inventario detectable en "
                       "este dataset: se omite la animacion en vez de generar "
                       "una con datos sinteticos.")

    # ============================================================
    # Innovacion - Capa 10: War Room de escenarios estrategicos
    # Solo visible en dominio Estrategia (o General, con aviso).
    # ============================================================
    if active_domain in ("estrategia", "general"):
        st.divider()
        st.subheader("🧭 Innovacion — War Room de escenarios")
        if active_domain == "general":
            st.warning("Este modulo esta pensado para el dominio Estrategia. "
                      "Puedes usarlo igual, pero interpreta los resultados con "
                      "cautela fuera de ese contexto.")
        st.caption("Los sliders re-predicen con el modelo YA entrenado en la "
                  "Capa 3 (no se reentrena nada aqui). La matriz BCG usa los "
                  "clusters YA calculados por el Agente No Supervisado.")

        war_df: pd.DataFrame = st.session_state.get("clean_df")
        sup_result = st.session_state.get("analysis_results", {}).get("supervised")
        uns_result = st.session_state.get("analysis_results", {}).get("unsupervised")
        target_wr = st.session_state.get("diagnosis", {}).get("target_column")

        wr_tab_sliders, wr_tab_bcg = st.tabs(
            ["🎛️ Escenarios what-if", "📊 Matriz BCG"])

        # ---------------- Tab 1: sliders what-if ----------------
        with wr_tab_sliders:
            if not sup_result or sup_result.get("status") != "ok" or "trained_model" not in sup_result:
                st.info("Ejecuta el analisis Supervisado (Capa 3) primero para "
                       "habilitar los escenarios what-if.")
            elif war_df is None:
                st.info("Carga y limpia un dataset primero.")
            else:
                importance = sup_result.get("numeric_feature_importance", {})
                if not importance:
                    st.info("El modelo entrenado no expone variables numericas "
                           "con importancia para construir sliders.")
                else:
                    baseline = build_baseline_row(war_df, target_wr)
                    overrides: dict[str, float] = {}
                    st.caption("Sliders sobre las variables mas importantes "
                              "segun el grafico de importancia del Agente "
                              "Supervisado:")
                    for col, imp in importance.items():
                        col_series = war_df[col]
                        lo_v, hi_v = float(col_series.min()), float(col_series.max())
                        default_v = float(baseline[col]) if col in baseline.index else lo_v
                        overrides[col] = st.slider(
                            f"{col} (importancia {imp:.2f})",
                            min_value=lo_v, max_value=hi_v, value=default_v,
                            key=_next_key(f"wr_slider_{col}"))

                    scenario = predict_scenario(sup_result, baseline, overrides)
                    if scenario.get("status") == "ok":
                        if sup_result.get("task") == "clasificacion":
                            st.success(f"**Prediccion del escenario:** "
                                      f"{scenario.get('predicted_label')}")
                            if scenario.get("proba"):
                                proba_df = pd.DataFrame({
                                    "Clase": list(scenario["proba"].keys()),
                                    "Probabilidad": list(scenario["proba"].values()),
                                })
                                st.dataframe(proba_df, width="stretch", hide_index=True)
                        else:
                            st.success(f"**Prediccion del escenario para "
                                      f"'{target_wr}':** {scenario['raw_prediction']:,.2f}")

                        if st.button("💾 Guardar este escenario para comparar",
                                    key=_next_key("wr_save_scenario")):
                            saved = st.session_state.get("wr_scenarios", [])
                            saved.append({"overrides": dict(overrides),
                                         "result": scenario})
                            st.session_state.wr_scenarios = saved[-3:]  # maximo 3
                            st.rerun()
                    else:
                        st.warning(scenario.get("reason", "No se pudo re-predecir."))

                    saved_scenarios = st.session_state.get("wr_scenarios", [])
                    if saved_scenarios:
                        st.markdown("**Comparador de escenarios guardados**")
                        comp_cols = st.columns(len(saved_scenarios))
                        for col_ui, sc in zip(comp_cols, saved_scenarios):
                            with col_ui:
                                with st.container(border=True):
                                    for k, v in sc["overrides"].items():
                                        st.caption(f"{k} = {v:.2f}")
                                    res = sc["result"]
                                    if sup_result.get("task") == "clasificacion":
                                        st.metric("Prediccion", res.get("predicted_label"))
                                    else:
                                        st.metric("Prediccion",
                                                 f"{res.get('raw_prediction', 0):,.2f}")
                        if st.button("🗑️ Limpiar comparador",
                                    key=_next_key("wr_clear_scenarios")):
                            st.session_state.wr_scenarios = []
                            st.rerun()

        # ---------------- Tab 2: matriz BCG ----------------
        with wr_tab_bcg:
            if not uns_result or uns_result.get("status") != "ok":
                st.info("Ejecuta el analisis No Supervisado (Capa 3) primero "
                       "para habilitar la matriz BCG.")
            else:
                overall_mean = uns_result.get("overall_mean", {})
                if len(overall_mean) < 2:
                    st.info("No hay suficientes columnas numericas para armar "
                           "una matriz BCG (se necesitan al menos 2 ejes).")
                else:
                    axis_cols = list(overall_mean.keys())
                    default_x, default_y = default_bcg_axes(overall_mean, target_wr)
                    c1, c2 = st.columns(2)
                    with c1:
                        x_axis = st.selectbox("Eje X (ej. participacion)",
                                              options=axis_cols,
                                              index=axis_cols.index(default_x)
                                              if default_x in axis_cols else 0,
                                              key=_next_key("wr_bcg_x"))
                    with c2:
                        y_axis = st.selectbox("Eje Y (ej. crecimiento)",
                                              options=axis_cols,
                                              index=axis_cols.index(default_y)
                                              if default_y in axis_cols else 0,
                                              key=_next_key("wr_bcg_y"))

                    quadrants = build_bcg_quadrants(uns_result, x_axis, y_axis)
                    if quadrants:
                        st.plotly_chart(bcg_matrix_scatter(quadrants, x_axis, y_axis),
                                        width="stretch", key=_next_key("wr_bcg_chart"))
                        st.caption("Cuadrante asignado comparando cada cluster "
                                  "contra la media global real de esas columnas "
                                  "(sin umbrales inventados).")
                        with st.spinner("Resumiendo la matriz..."):
                            summary = summarize_bcg_with_llm(quadrants, x_axis, y_axis,
                                                             active_domain)
                        st.info(summary)
                    else:
                        st.warning("No se pudo construir la matriz con estos ejes.")
