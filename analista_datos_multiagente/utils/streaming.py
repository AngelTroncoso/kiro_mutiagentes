"""Infraestructura de streaming entre los agentes (productores) y la UI (consumidor).

Los agentes corren dentro del grafo de LangGraph, en un hilo aparte del hilo de
Streamlit. Todo lo que producen (narracion token a token, metricas, graficos) se
publica en un `EventBus`, y la UI lo consume como un generador de eventos.

Contrato de eventos que llegan a la UI (`EventBus.stream()`):

    {"type": "agent_start",  "agent": str, "title": str, "icon": str}
    {"type": "narration",    "agent": str, "stream": Iterator[str]}   # para st.write_stream
    {"type": "text",         "agent": str, "text": str, "style": str}
    {"type": "result",       "agent": str, "key": str, "payload": dict}
    {"type": "chart",        "agent": str, "title": str, "figure": go.Figure, "caption": str}
    {"type": "table",        "agent": str, "title": str, "data": pd.DataFrame}
    {"type": "warning",      "agent": str, "text": str}
    {"type": "agent_end",    "agent": str, "summary": str}
    {"type": "error",        "agent": str, "text": str, "traceback": str}

El evento `narration` entrega un sub-generador de tokens: el consumidor DEBE
drenarlo (por ejemplo con `st.write_stream`) antes de pedir el siguiente evento.
Si no lo drena, los tokens se descartan de forma segura.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable, Iterator

_SENTINEL = "__pipeline_end__"


class EventBus:
    """Cola de eventos thread-safe entre los agentes y la interfaz."""

    def __init__(self) -> None:
        self._q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._pending: deque[dict[str, Any]] = deque()
        self._closed = False
        self.transcript: list[str] = []

    # ------------------------------------------------------------------
    # Lado productor (agentes)
    # ------------------------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        self._q.put(event)

    def agent_start(self, agent: str, title: str, icon: str = "*") -> None:
        self.emit({"type": "agent_start", "agent": agent, "title": title, "icon": icon})

    def agent_end(self, agent: str, summary: str = "") -> None:
        self.emit({"type": "agent_end", "agent": agent, "summary": summary})

    def text(self, agent: str, text: str, style: str = "markdown") -> None:
        self.emit({"type": "text", "agent": agent, "text": text, "style": style})

    def warning(self, agent: str, text: str) -> None:
        self.emit({"type": "warning", "agent": agent, "text": text})

    def result(self, agent: str, key: str, payload: dict[str, Any]) -> None:
        self.emit({"type": "result", "agent": agent, "key": key, "payload": payload})

    def chart(self, agent: str, title: str, figure: Any, caption: str = "") -> None:
        self.emit(
            {"type": "chart", "agent": agent, "title": title, "figure": figure, "caption": caption}
        )

    def table(self, agent: str, title: str, data: Any) -> None:
        self.emit({"type": "table", "agent": agent, "title": title, "data": data})

    def error(self, agent: str, text: str, tb: str = "") -> None:
        self.emit({"type": "error", "agent": agent, "text": text, "traceback": tb})

    def narrate(self, agent: str, tokens: Iterable[str]) -> str:
        """Publica una narracion token a token y devuelve el texto completo."""
        self.emit({"type": "narration_start", "agent": agent})
        chunks: list[str] = []
        try:
            for token in tokens:
                if not token:
                    continue
                chunks.append(token)
                self.emit({"type": "token", "agent": agent, "text": token})
        finally:
            self.emit({"type": "narration_end", "agent": agent})
        full = "".join(chunks)
        self.transcript.append(f"[{agent}] {full}")
        return full

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.emit({"type": _SENTINEL})

    # ------------------------------------------------------------------
    # Lado consumidor (Streamlit)
    # ------------------------------------------------------------------
    def stream(self, is_alive: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
        """Genera los eventos en orden. Convierte narraciones en sub-generadores."""
        while True:
            event = self._next(is_alive)
            if event is None:  # el productor murio sin cerrar el bus
                return
            kind = event.get("type")
            if kind == _SENTINEL:
                return
            if kind == "narration_start":
                yield {
                    "type": "narration",
                    "agent": event.get("agent", ""),
                    "stream": self._token_stream(is_alive),
                }
            elif kind in ("token", "narration_end"):
                continue  # sub-stream no consumido: se descarta sin romper el orden
            else:
                yield event

    def _next(self, is_alive: Callable[[], bool] | None) -> dict[str, Any] | None:
        if self._pending:
            return self._pending.popleft()
        while True:
            try:
                return self._q.get(timeout=0.1)
            except queue.Empty:
                if is_alive is not None and not is_alive():
                    # Damos una ultima pasada por si quedaron eventos encolados.
                    try:
                        return self._q.get_nowait()
                    except queue.Empty:
                        return None

    def _token_stream(self, is_alive: Callable[[], bool] | None) -> Iterator[str]:
        while True:
            event = self._next(is_alive)
            if event is None:
                return
            kind = event.get("type")
            if kind == "token":
                yield event["text"]
            elif kind == "narration_end":
                return
            else:
                # Evento fuera de lugar: lo devolvemos a la cola logica.
                self._pending.append(event)
                return


def run_in_thread(target: Callable[[], Any], bus: EventBus) -> threading.Thread:
    """Ejecuta el pipeline en un hilo y garantiza el cierre del bus."""

    def _runner() -> None:
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - se reporta a la UI
            import traceback

            bus.error("orquestador", f"El pipeline se detuvo: {exc}", traceback.format_exc())
        finally:
            bus.close()

    thread = threading.Thread(target=_runner, name="pipeline", daemon=True)
    thread.start()
    return thread


def typewriter(text: str, chunk_size: int = 3, delay: float = 0.012) -> Iterator[str]:
    """Convierte texto ya generado en un stream de tokens (modo sin LLM)."""
    words = text.split(" ")
    buffer: list[str] = []
    for word in words:
        buffer.append(word)
        if len(buffer) >= chunk_size:
            yield " ".join(buffer) + " "
            buffer = []
            if delay:
                time.sleep(delay)
    if buffer:
        yield " ".join(buffer)
