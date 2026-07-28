"""Capa 0 - Orquestacion del pipeline multiagente con LangGraph.

El pipeline se divide en DOS fases porque hay una interaccion humana en el medio
(el usuario elige que analisis ejecutar tras el diagnostico del router):

    Fase 1 (diagnostico):   ETL  ->  Router  ->  [pausa: el usuario decide]
    Fase 2 (analisis):      dispatch --(condicional)-->  {supervisado,
                            no_supervisado, refuerzo}  ->  recomendador  ->
                            reporte_ejecutivo

La ramificacion condicional de la Fase 2 se modela con `add_conditional_edges`:
segun `selected_analyses`, el grafo enruta a los agentes elegidos y luego converge
en el recomendador. Cada agente escribe una clave distinta del estado, por lo que
no hay conflictos de escritura concurrente.

Si LangGraph no estuviera disponible, hay un `SequentialRunner` de respaldo que
ejecuta los mismos nodos en orden, con identica firma.
"""

from __future__ import annotations

from typing import Any, Callable

from agents.etl_agent import run_etl
from agents.executive_report_agent import run_executive_report
from agents.recommender_agent import run_recommender
from agents.reinforcement_agent import run_reinforcement
from agents.router_agent import run_router
from agents.supervised_agent import run_supervised
from agents.unsupervised_agent import run_unsupervised

from .state import PipelineState

# Mapa enfoque -> (nombre de nodo, funcion).
ANALYSIS_NODES: dict[str, tuple[str, Callable]] = {
    "supervised": ("supervised", run_supervised),
    "unsupervised": ("unsupervised", run_unsupervised),
    "reinforcement": ("reinforcement", run_reinforcement),
}


def _try_langgraph():
    try:
        from langgraph.graph import END, START, StateGraph

        return StateGraph, START, END
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# FASE 1: diagnostico (ETL -> Router)
# ----------------------------------------------------------------------
def run_diagnosis_phase(state: PipelineState) -> PipelineState:
    """Ejecuta ETL y Router, devolviendo el estado enriquecido."""
    lg = _try_langgraph()
    if lg is None:
        return _sequential_diagnosis(state)

    StateGraph, START, END = lg
    g = StateGraph(PipelineState)
    g.add_node("etl", _wrap(run_etl))
    g.add_node("router", _wrap(run_router))
    g.add_edge(START, "etl")
    g.add_edge("etl", "router")
    g.add_edge("router", END)
    app = g.compile()
    result = app.invoke(state)
    return result  # type: ignore[return-value]


def _sequential_diagnosis(state: PipelineState) -> PipelineState:
    state.update(_wrap(run_etl)(state))  # type: ignore[arg-type]
    state.update(_wrap(run_router)(state))  # type: ignore[arg-type]
    return state


# ----------------------------------------------------------------------
# FASE 2: analisis seleccionado + recomendacion
# ----------------------------------------------------------------------
def run_analysis_phase(state: PipelineState) -> PipelineState:
    """Ejecuta los agentes seleccionados (ramificacion condicional) y recomienda."""
    selected = [s for s in state.get("selected_analyses", []) if s in ANALYSIS_NODES]
    lg = _try_langgraph()
    if lg is None:
        return _sequential_analysis(state, selected)

    StateGraph, START, END = lg
    g = StateGraph(PipelineState)

    # Nodo dispatcher (no-op) desde el que ramificamos condicionalmente.
    g.add_node("dispatch", lambda s: {})
    for key in ANALYSIS_NODES:
        node_name, fn = ANALYSIS_NODES[key]
        g.add_node(node_name, _wrap(fn))
    g.add_node("recommender", _wrap(run_recommender))
    g.add_node("executive_report", _wrap(run_executive_report))

    g.add_edge(START, "dispatch")

    # Ramificacion condicional: elegimos a que nodos ir segun la seleccion.
    def route(s: PipelineState) -> list[str]:
        chosen = [ANALYSIS_NODES[k][0] for k in s.get("selected_analyses", [])
                  if k in ANALYSIS_NODES]
        return chosen or ["recommender"]

    possible = [ANALYSIS_NODES[k][0] for k in ANALYSIS_NODES] + ["recommender"]
    g.add_conditional_edges("dispatch", route, possible)

    # Todos los agentes convergen en el recomendador, y este en el reporte ejecutivo.
    for key in ANALYSIS_NODES:
        node_name, _ = ANALYSIS_NODES[key]
        g.add_edge(node_name, "recommender")
    g.add_edge("recommender", "executive_report")
    g.add_edge("executive_report", END)

    app = g.compile()
    result = app.invoke(state)
    return result  # type: ignore[return-value]


def _sequential_analysis(state: PipelineState, selected: list[str]) -> PipelineState:
    for key in selected:
        _, fn = ANALYSIS_NODES[key]
        state.update(_wrap(fn)(state))  # type: ignore[arg-type]
    state.update(_wrap(run_recommender)(state))  # type: ignore[arg-type]
    state.update(_wrap(run_executive_report)(state))  # type: ignore[arg-type]
    return state


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
# Clave de resultado que escribe cada agente, para degradar con elegancia.
_RESULT_KEY = {
    "run_supervised": "supervised_result",
    "run_unsupervised": "unsupervised_result",
    "run_reinforcement": "reinforcement_result",
    "run_recommender": "recommendation",
    "run_executive_report": "executive_report",
}


def _wrap(fn: Callable[[PipelineState], dict[str, Any]]):
    """Adapta un run_X(state)->dict a un nodo de LangGraph.

    Aisla el fallo de un agente: si uno lanza una excepcion, se reporta a la UI y
    el nodo devuelve un resultado 'failed' en su clave, sin tumbar el resto del
    grafo (los demas agentes y el recomendador siguen funcionando).
    """

    def _node(state: PipelineState) -> dict[str, Any]:
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001
            import traceback

            bus = state.get("bus")
            if bus is not None:
                bus.error(fn.__name__, f"El agente fallo: {exc}",
                          traceback.format_exc())
            key = _RESULT_KEY.get(fn.__name__)
            if key:
                return {key: {"status": "failed", "reason": str(exc)}}
            return {}

    return _node


def run_full_pipeline(state: PipelineState) -> PipelineState:
    """Corre todo de una vez (util para pruebas headless sin interaccion humana)."""
    state = run_diagnosis_phase(state)
    if not state.get("selected_analyses"):
        state["selected_analyses"] = list(state.get("diagnosis", {}).get("applicable", []))
    state = run_analysis_phase(state)
    return state
