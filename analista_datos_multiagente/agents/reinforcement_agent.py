"""Capa 3 - Agente por Refuerzo (aproximacion: contextual bandit).

NOTA DE DISENO (declarada al usuario en la narracion):
El aprendizaje por refuerzo clasico necesita un entorno con estados, acciones y
recompensas que evolucionan al interactuar. Un dataset tabular estatico NO es eso.
Por eso construimos un ENTORNO SIMULADO:

    - Cada fila del dataset es un "contexto" (estado observado).
    - Las "acciones" son las posibles categorias/rangos de la variable objetivo.
    - La "recompensa" es 1 si el agente elige la accion correcta para esa fila, 0 si no.

Esto es un *contextual bandit*: el agente aprende una politica que mapea contexto
-> accion para maximizar la recompensa. Es la aproximacion honesta y explicable de
RL sobre datos tabulares, sin fingir un entorno interactivo real.

Implementacion: LinUCB (Linear Upper Confidence Bound), un algoritmo de bandit
contextual clasico, ligero y sin dependencias pesadas. Si Gymnasium esta instalado
se formaliza el entorno; si no, el bucle de aprendizaje funciona igual.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import visualization as viz
from utils.llm import narrate_stream

from .common import build_features, encode_target

AGENT = "Refuerzo"


class LinUCB:
    """Contextual bandit lineal con exploracion optimista (UCB)."""

    def __init__(self, n_actions: int, n_features: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_actions)]

    def select(self, x: np.ndarray) -> int:
        x = x.reshape(-1, 1)
        best_a, best_p = 0, -np.inf
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            # (theta.T @ x) y (x.T @ A_inv @ x) son matrices 1x1: extraemos el escalar.
            mean = float((theta.T @ x).item())
            bonus = self.alpha * float(np.sqrt((x.T @ A_inv @ x).item()))
            p = mean + bonus
            if p > best_p:
                best_p, best_a = p, a
        return best_a

    def update(self, a: int, x: np.ndarray, reward: float) -> None:
        x = x.reshape(-1, 1)
        self.A[a] += x @ x.T
        self.b[a] += reward * x


def _discretize_continuous(y: np.ndarray, n_bins: int = 3) -> tuple[np.ndarray, list[str]]:
    """Convierte un objetivo continuo en rangos (bajo/medio/alto)."""
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    codes = np.digitize(y, edges[1:-1])
    names = ["bajo", "medio", "alto"][:n_bins] if n_bins <= 3 else \
        [f"rango {i+1}" for i in range(n_bins)]
    return codes, names


def run_reinforcement(state) -> dict[str, Any]:
    bus = state["bus"]
    df: pd.DataFrame = state["clean_df"]
    diagnosis = state.get("diagnosis", {})
    target = state.get("target_column") or diagnosis.get("target_column")

    bus.agent_start(AGENT, "Agente por Refuerzo - aprender a decidir", icon="🕹️")

    if not target or target not in df.columns:
        bus.warning(AGENT, "Sin variable objetivo no se puede definir la recompensa.")
        bus.agent_end(AGENT, "Omitido: sin objetivo.")
        return {"reinforcement_result": {"status": "skipped", "reason": "sin objetivo"}}

    # Declaracion honesta de la limitacion (criterio de aceptacion).
    disclaimer = (
        f"Importante: el refuerzo clasico necesita un entorno donde el agente actua y "
        f"recibe recompensas. Aqui tengo un dataset estatico, asi que construyo un "
        f"entorno SIMULADO: cada fila es una situacion, las acciones son los posibles "
        f"valores de '{target}', y gano recompensa (1) cuando acierto la accion correcta. "
        f"Esto se llama 'contextual bandit' y es una aproximacion honesta de refuerzo "
        f"sobre datos tabulares, no un entorno interactivo real."
    )
    bus.narrate(AGENT, narrate_stream(
        "Explica a un principiante, con honestidad, por que el refuerzo sobre un dataset "
        f"estatico es una aproximacion (contextual bandit). Contexto: {disclaimer}",
        disclaimer))

    # --- Construir contexto (X) y acciones (y discreto) ---
    y_raw = df[target]
    y_enc, labels = encode_target(y_raw)
    if labels is None:  # objetivo continuo -> discretizar en rangos
        y_enc, labels = _discretize_continuous(np.asarray(y_enc, dtype=float))
    n_actions = int(len(np.unique(y_enc)))
    if n_actions < 2:
        bus.warning(AGENT, "El objetivo no tiene suficiente variedad de acciones.")
        bus.agent_end(AGENT, "Omitido: acciones insuficientes.")
        return {"reinforcement_result": {"status": "skipped", "reason": "acciones insuficientes"}}

    feats = build_features(df, exclude=[target])
    X = StandardScaler().fit_transform(feats.X)
    # Sesgo (intercepto) como columna extra.
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    y_enc = np.asarray(y_enc, dtype=int)

    _try_gymnasium_note(bus)

    # --- Bucle de aprendizaje online (varias pasadas sobre las filas) ---
    rng = np.random.default_rng(42)
    agent = LinUCB(n_actions=n_actions, n_features=X.shape[1], alpha=0.8)
    n = X.shape[0]
    n_steps = min(4000, n * 3)
    rewards: list[float] = []
    for t in range(n_steps):
        i = int(rng.integers(0, n))
        a = agent.select(X[i])
        r = 1.0 if a == y_enc[i] else 0.0
        agent.update(a, X[i], r)
        rewards.append(r)

    # Baseline: elegir al azar.
    baseline = 1.0 / n_actions
    # Recompensa media en la ultima ventana (politica ya aprendida).
    window = max(50, n_steps // 5)
    final_reward = float(np.mean(rewards[-window:]))

    bus.chart(AGENT, "Curva de aprendizaje",
              viz.reward_curve(rewards, baseline=baseline),
              caption="La recompensa media deberia superar a la eleccion al azar (linea naranja).")

    lift = final_reward - baseline
    quality = ("fuerte" if lift > 0.3 else "moderada" if lift > 0.1 else "leve")
    fallback = (
        f"El agente aprendio una politica de decision con acierto final "
        f"{final_reward*100:.1f}%, frente a {baseline*100:.1f}% eligiendo al azar "
        f"(mejora {quality}). Esto muestra que el contexto de cada fila SI ayuda a "
        f"decidir el valor de '{target}'. Recuerda: es una simulacion tipo bandit, "
        f"util para entender el potencial de una estrategia de decision basada en datos."
    )
    bus.narrate(AGENT, narrate_stream(
        "Explica a un principiante que aprendio el agente de refuerzo y como leer la "
        f"curva de recompensa. Resultado: {fallback}", fallback))

    result = {
        "status": "ok",
        "approach": "contextual_bandit_linucb",
        "n_actions": n_actions,
        "actions": labels,
        "final_reward": final_reward,
        "baseline": baseline,
        "lift": lift,
        "quality": quality,
        "score_norm": float(np.clip(final_reward, 0, 1)),
    }
    bus.result(AGENT, "reinforcement", result)
    bus.agent_end(AGENT, f"Acierto {final_reward*100:.1f}% vs azar {baseline*100:.1f}%.")
    return {"reinforcement_result": result}


def _try_gymnasium_note(bus) -> None:
    """Si Gymnasium esta disponible, lo mencionamos (formalizacion del entorno)."""
    try:
        import gymnasium  # noqa: F401
        bus.text(AGENT, "_Entorno formalizable con Gymnasium (espacio de observacion = "
                        "contexto, espacio de accion = valores del objetivo)._", style="caption")
    except Exception:  # noqa: BLE001
        pass
