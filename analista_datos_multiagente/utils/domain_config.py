"""Capa 0 - Configuracion de dominios de negocio (Finanzas / Logistica / Estrategia).

Este modulo es PURAMENTE de presentacion. Ninguna funcion aqui calcula metricas ni
transforma datos: solo provee vocabulario, textos curados y colores para que ETL,
Router, Recomendador y Reporte Ejecutivo hablen "el idioma" del negocio elegido.

Regla de oro (no negociable, ver system prompt del proyecto): el dominio NUNCA
cambia una cifra, un modelo entrenado o un score. Cambiar de dominio sobre el mismo
dataset debe reproducir exactamente las mismas metricas; solo cambia el marco con
el que se presentan.

Las claves de `business_goals` de cada dominio son SIEMPRE las 4 claves abstractas
que ya usa `recommender_agent.GOAL_WEIGHTS` ("predecir", "segmentar", "decidir",
"explorar"). Eso garantiza que los pesos y la matematica del recomendador no se
toquen: el dominio solo reemplaza el TEXTO que ve el usuario para cada opcion.
"""

from __future__ import annotations

from typing import Any

# Vocabulario neutro: reproduce la redaccion original de la app cuando no hay un
# dominio de negocio real seleccionado (fallback seguro, nunca se muestra en el
# selector de la UI).
NEUTRAL_VOCAB: dict[str, str] = {
    "row_word": "fila",
    "null_phrase": "valores nulos",
    "duplicate_phrase": "filas duplicadas",
    "outlier_phrase": "valores atipicos (outliers)",
    "target_phrase": "variable objetivo",
    "group_word": "grupo",
    "action_word": "accion",
}

_NEUTRAL_GOALS: dict[str, str] = {
    "predecir": "Predecir un valor o categoria concreta",
    "segmentar": "Descubrir grupos o segmentos",
    "decidir": "Optimizar una decision o accion",
    "explorar": "Explorar el dataset sin objetivo fijo",
}

DOMAINS: dict[str, dict[str, Any]] = {
    "finanzas": {
        "key": "finanzas",
        "label": "Finanzas",
        "icon": "💰",
        "tagline": "Riesgo, retencion de ingresos y deteccion de fraude.",
        "accent_primary": "#10B981",   # esmeralda
        "accent_secondary": "#F5B700",  # dorado
        "vocab": {
            "row_word": "cliente o transaccion",
            "null_phrase": "datos faltantes en el historial financiero",
            "duplicate_phrase": "transacciones registradas mas de una vez",
            "outlier_phrase": "movimientos con montos atipicos (posibles anomalias)",
            "target_phrase": "indicador de riesgo financiero",
            "group_word": "segmento de clientes",
            "action_word": "politica de credito o retencion",
        },
        "domain_keywords": [
            "cliente", "transaccion", "transacción", "monto", "credito", "crédito",
            "pago", "factura", "riesgo", "fraude", "saldo", "ingreso", "mora",
            "default", "churn", "deuda", "cartera", "prestamo", "préstamo",
        ],
        "target_frames": {
            "riesgo de fuga de ingresos": ["churn", "fuga", "cancel", "abandono", "baja"],
            "riesgo de incumplimiento crediticio": ["default", "mora", "impago", "atraso"],
            "indicador de fraude o anomalia": ["fraude", "anomal"],
        },
        "business_goals": {
            "predecir": "Minimizar riesgo crediticio",
            "segmentar": "Detectar fraude o anomalias",
            "decidir": "Maximizar retencion de ingresos",
            "explorar": "Explorar el comportamiento financiero sin objetivo fijo",
        },
        "kpi_context": (
            "ratios financieros, probabilidad de incumplimiento (default), "
            "exposicion al riesgo y retencion de ingresos (revenue retention)"
        ),
        "sample_dataset": "datos_ejemplo_finanzas.csv",
    },
    "logistica": {
        "key": "logistica",
        "label": "Logistica",
        "icon": "🚚",
        "tagline": "Tiempos de entrega, inventario y costo de transporte.",
        "accent_primary": "#22D3EE",   # cian
        "accent_secondary": "#FB923C",  # naranjo
        "vocab": {
            "row_word": "pedido o envio",
            "null_phrase": "datos faltantes en el registro logistico",
            "duplicate_phrase": "pedidos registrados mas de una vez",
            "outlier_phrase": "tiempos o costos fuera de lo normal (posibles cuellos de botella)",
            "target_phrase": "indicador de cumplimiento de entrega",
            "group_word": "grupo de rutas o SKUs",
            "action_word": "politica de inventario o ruteo",
        },
        "domain_keywords": [
            "pedido", "envio", "envío", "entrega", "ruta", "inventario", "stock",
            "sku", "lead_time", "leadtime", "transporte", "flota", "almacen",
            "almacén", "retraso", "delay", "distancia", "peso", "volumen",
        ],
        "target_frames": {
            "incumplimiento de SLA": ["retraso", "delay", "atraso", "incumpl", "sla"],
            "riesgo de quiebre de inventario": ["stock", "inventario", "quiebre", "stockout"],
        },
        "business_goals": {
            "predecir": "Reducir tiempos de entrega",
            "segmentar": "Optimizar niveles de inventario",
            "decidir": "Minimizar costos de transporte",
            "explorar": "Explorar la operacion logistica sin objetivo fijo",
        },
        "kpi_context": (
            "lead time, cumplimiento de SLA, OTIF (on-time in-full) y costo por envio"
        ),
        "sample_dataset": "datos_ejemplo_logistica.csv",
    },
    "estrategia": {
        "key": "estrategia",
        "label": "Estrategia",
        "icon": "🧭",
        "tagline": "Crecimiento, segmentos de valor y asignacion de recursos.",
        "accent_primary": "#8B5CF6",   # violeta
        "accent_secondary": "#E879F9",  # magenta
        "vocab": {
            "row_word": "registro de mercado o segmento",
            "null_phrase": "datos faltantes en la informacion de mercado",
            "duplicate_phrase": "registros de mercado repetidos",
            "outlier_phrase": "valores fuera de lo esperado (posibles oportunidades o riesgos)",
            "target_phrase": "KPI estrategico",
            "group_word": "segmento estrategico",
            "action_word": "asignacion de recursos o inversion",
        },
        "domain_keywords": [
            "segmento", "mercado", "region", "región", "kpi", "ventas", "crecimiento",
            "canal", "producto", "categoria", "categoría", "cuota", "penetracion",
            "penetración", "competidor", "marca",
        ],
        "target_frames": {
            "KPI de crecimiento": ["crecimiento", "growth"],
            "segmento de alto valor": ["segmento", "valor", "tier"],
        },
        "business_goals": {
            "predecir": "Priorizar crecimiento",
            "segmentar": "Identificar segmentos de alto valor",
            "decidir": "Optimizar asignacion de recursos",
            "explorar": "Explorar el panorama estrategico sin objetivo fijo",
        },
        "kpi_context": (
            "posicionamiento de mercado, tasa de crecimiento, participacion de "
            "mercado (market share) y retorno de la inversion por segmento"
        ),
        "sample_dataset": "datos_ejemplo_estrategia.csv",
    },
    # Fallback neutro: NO se muestra en el selector de la UI. Se usa solo si el
    # estado no trae un dominio valido, para no romper compatibilidad.
    "general": {
        "key": "general",
        "label": "General",
        "icon": "🧪",
        "tagline": "Sin dominio de negocio especifico.",
        "accent_primary": "#8B5CF6",
        "accent_secondary": "#22D3EE",
        "vocab": NEUTRAL_VOCAB,
        "domain_keywords": [],
        "target_frames": {},
        "business_goals": dict(_NEUTRAL_GOALS),
        "kpi_context": "metricas generales de desempeño",
        "sample_dataset": None,
    },
}

# Los 3 dominios reales que se ofrecen en el selector de la interfaz.
DOMAIN_KEYS: list[str] = ["finanzas", "logistica", "estrategia"]


def get_domain(domain: str | None) -> dict[str, Any]:
    """Devuelve la configuracion del dominio, o el fallback neutro si no es valido."""
    key = (domain or "").strip().lower()
    return DOMAINS.get(key, DOMAINS["general"])


def vocab_for(domain: str | None) -> dict[str, str]:
    return get_domain(domain)["vocab"]


def goal_labels_for(domain: str | None) -> dict[str, str]:
    """Etiquetas de `business_goal` curadas por dominio (mismas claves abstractas)."""
    return get_domain(domain)["business_goals"]


def is_domain_match(columns: list[str], domain: str | None) -> bool:
    """Heuristica: ¿el dataset tiene columnas tipicas de este dominio?

    Solo mira los NOMBRES de columnas ya detectadas; no inventa ni infiere datos.
    Si el dominio es el neutro ("general"), se considera siempre coherente.
    """
    cfg = get_domain(domain)
    keywords = cfg.get("domain_keywords", [])
    if not keywords:
        return True
    cols_lower = [str(c).lower() for c in columns]
    return any(kw in col for kw in keywords for col in cols_lower)


def frame_target(target: str | None, domain: str | None) -> str:
    """Enmarca el nombre de la variable objetivo con el lenguaje del dominio.

    Solo reformula el NOMBRE de una columna que ya existe en el dataset; nunca
    inventa una columna ni cambia lo que el Router detecto matematicamente.
    """
    cfg = get_domain(domain)
    if not target:
        return cfg["vocab"]["target_phrase"]
    tname = str(target).lower()
    for frame, keywords in cfg.get("target_frames", {}).items():
        if any(kw in tname for kw in keywords):
            return frame
    return cfg["vocab"]["target_phrase"]
