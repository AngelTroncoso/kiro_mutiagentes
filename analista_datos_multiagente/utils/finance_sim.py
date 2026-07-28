"""Capa 8 - Simulador de flujo de caja / proyeccion de inversion (Monte Carlo).

Activo SOLO en el dominio Finanzas (ver app.py). Este modulo es matematica pura:
no narra con LLM ni decide nada de negocio, solo simula.

Principio no negociable (igual que el resto de la app): todo parametro de las
distribuciones de entrada/salida de caja es la media y desviacion estandar REALES
de una columna del dataset (`derive_distribution`), o un valor que el usuario fija
explicitamente en la UI. Nunca se inventa un supuesto oculto.

Conexion con el Recomendador (capa 4) / Supervisado (capa 3): si el enfoque
ganador fue supervisado y la variable objetivo es binaria (ej. churn), se puede
aplicar un ajuste que reduce el flujo de entrada en la proporcion REAL de casos
positivos ya detectada en `clean_df` (no una probabilidad inventada por fila).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Columnas candidatas a "flujo de caja" segun el vocabulario financiero ya usado
# en domain_config.py. Es solo una sugerencia por defecto; el usuario puede
# mapear cualquier columna numerica manualmente en la UI.
_MONETARY_KEYWORDS = (
    "monto", "ingreso", "pago", "factura", "saldo", "credito", "crédito",
    "deuda", "cartera", "prestamo", "préstamo", "transaccion", "transacción",
    "precio", "venta", "flujo", "caja", "capital",
)


def detect_monetary_columns(df: pd.DataFrame) -> list[str]:
    """Columnas numericas cuyo nombre sugiere que representan dinero.

    Heuristica de solo-nombres (no inventa datos): si ninguna columna calza,
    devuelve la lista vacia y la UI debe ofrecer todas las numericas para que
    el usuario elija manualmente.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = []
    for col in numeric_cols:
        name = str(col).lower()
        if any(kw in name for kw in _MONETARY_KEYWORDS):
            out.append(col)
    return out


def derive_distribution(series: pd.Series) -> dict[str, float]:
    """Media y desviacion estandar REALES de una columna (sin inventar nada)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def derive_split_distribution(series: pd.Series) -> dict[str, Any]:
    """Si la columna mezcla signos (ej. movimientos +/-), la separa en

    entradas (valores positivos) y salidas (valor absoluto de los negativos).
    Si no mezcla signos, todo el valor se trata como entrada.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    pos = s[s > 0]
    neg = s[s < 0]
    if len(pos) > 0 and len(neg) > 0:
        mode = "split_por_signo"
        inflow = derive_distribution(pos)
        outflow = derive_distribution(neg.abs())
    else:
        mode = "solo_entradas"
        inflow = derive_distribution(s.abs())
        outflow = {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mode": mode, "inflow": inflow, "outflow": outflow}


def detect_binary_rate(df: pd.DataFrame, target_col: str | None) -> float | None:
    """Si `target_col` es binaria (dos valores), devuelve la proporcion REAL

    de la clase codificada como "1"/positiva (ej. proporcion de churn=1). Se usa
    como ajuste trazable a `clean_df`, nunca como una probabilidad inventada.
    """
    if not target_col or target_col not in df.columns:
        return None
    s = df[target_col].dropna()
    uniques = s.unique()
    if len(uniques) != 2:
        return None
    # Intentamos identificar la clase "positiva" (1, True, o la etiqueta mas
    # comun asociada a riesgo: "si", "yes", "1").
    try:
        s_num = pd.to_numeric(s, errors="raise")
        positive_label = 1 if 1 in s_num.unique() else s_num.max()
        rate = float((s_num == positive_label).mean())
    except (ValueError, TypeError):
        s_str = s.astype(str).str.lower()
        positive_candidates = {"1", "true", "si", "sí", "yes", "fraude", "mora"}
        matches = s_str.isin(positive_candidates)
        if not matches.any():
            return None
        rate = float(matches.mean())
    return rate


@dataclass
class MonteCarloInputs:
    initial_capital: float
    horizon_periods: int
    discount_rate: float  # tasa de descuento POR PERIODO (ej. mensual si horizon es en meses)
    inflow_mean: float
    inflow_std: float
    outflow_mean: float
    outflow_std: float
    n_iterations: int = 8000
    seed: int = 42
    churn_adjustment_rate: float | None = None  # proporcion real aplicada a las entradas
    source_note: str = ""  # de donde vienen los parametros (trazabilidad para la UI)


@dataclass
class MonteCarloResult:
    inputs: MonteCarloInputs
    paths: np.ndarray                 # (n_iter, horizon+1) flujo acumulado sin descontar
    npv_samples: np.ndarray           # (n_iter,) VAN por iteracion
    npv_p10: float
    npv_p50: float
    npv_p90: float
    prob_flujo_negativo: float        # P(flujo acumulado final < 0)
    irr_representative: float | None  # TIR sobre el flujo PROMEDIO simulado
    mean_net_cashflows: list[float] = field(default_factory=list)


def compute_irr(cashflows: list[float]) -> float | None:
    """TIR: tasa r tal que NPV(r) = 0, via busqueda de raiz (Brent).

    `cashflows[0]` es el desembolso inicial (negativo por convencion) y el resto
    son los flujos netos periodo a periodo.
    """
    if len(cashflows) < 2:
        return None

    def npv_at(r: float) -> float:
        return sum(cf / (1.0 + r) ** t for t, cf in enumerate(cashflows))

    try:
        from scipy.optimize import brentq

        lo, hi = -0.999999, 10.0
        f_lo, f_hi = npv_at(lo), npv_at(hi)
        if f_lo * f_hi > 0:
            # No hay cambio de signo en el rango: la TIR no existe o esta fuera
            # de un rango razonable (ej. flujos siempre positivos o siempre
            # negativos). Devolvemos None en vez de forzar un numero.
            return None
        return float(brentq(npv_at, lo, hi))
    except Exception:
        return None


def run_monte_carlo(inputs: MonteCarloInputs) -> MonteCarloResult:
    rng = np.random.default_rng(inputs.seed)
    n = inputs.n_iterations
    T = inputs.horizon_periods

    inflow_mean = inputs.inflow_mean
    if inputs.churn_adjustment_rate:
        inflow_mean = inflow_mean * (1.0 - inputs.churn_adjustment_rate)

    inflow = rng.normal(inflow_mean, max(inputs.inflow_std, 1e-9), size=(n, T))
    outflow = rng.normal(inputs.outflow_mean, max(inputs.outflow_std, 1e-9), size=(n, T))
    # Los flujos no pueden ser negativos por definicion (son magnitudes de dinero
    # que entra/sale); recortamos la cola negativa de la distribucion normal.
    inflow = np.clip(inflow, 0, None)
    outflow = np.clip(outflow, 0, None)

    net = inflow - outflow  # (n, T)
    cum_net = np.cumsum(net, axis=1)
    paths = np.hstack([
        np.full((n, 1), inputs.initial_capital),
        inputs.initial_capital + cum_net,
    ])

    discount_factors = (1.0 + inputs.discount_rate) ** np.arange(1, T + 1)
    discounted_net = net / discount_factors
    npv_samples = -inputs.initial_capital + discounted_net.sum(axis=1)

    p10, p50, p90 = (float(x) for x in np.percentile(npv_samples, [10, 50, 90]))
    prob_negative = float((paths[:, -1] < 0).mean())

    mean_net = net.mean(axis=0).tolist()
    irr = compute_irr([-inputs.initial_capital, *mean_net])

    return MonteCarloResult(
        inputs=inputs,
        paths=paths,
        npv_samples=npv_samples,
        npv_p10=p10,
        npv_p50=p50,
        npv_p90=p90,
        prob_flujo_negativo=prob_negative,
        irr_representative=irr,
        mean_net_cashflows=mean_net,
    )
