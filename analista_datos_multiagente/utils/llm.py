"""Capa de narracion pedagogica sobre Groq (con fallback local determinista).

Objetivo de diseno:
- Si hay `GROQ_API_KEY`, cada agente narra en streaming real usando un LLM rapido.
- Si no hay clave o la libreria falla, el sistema NO se rompe: cae a un narrador
  local que convierte un texto base en un stream tipo maquina de escribir. Asi la
  app es 100% funcional para aprender aunque el usuario no tenga clave de Groq.

El prompt de sistema fija el tono pedagogico exigido en los criterios de
aceptacion: explicar en lenguaje simple, sin jerga innecesaria, y justificando
el "por que" de cada decision.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from .streaming import typewriter

# Modelo por defecto en Groq. Rapido, apto para streaming fluido.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "Eres un tutor experto en ciencia de datos que acompana a una persona sin "
    "conocimientos previos de Machine Learning. Explicas en espanol, con lenguaje "
    "simple y cercano, evitando jerga; cuando usas un termino tecnico, lo defines "
    "en una frase. Siempre justificas el POR QUE de cada decision. Eres conciso: "
    "2 a 5 frases, sin listas largas ni encabezados. Hablas en primera persona "
    "como el agente que esta ejecutando el paso."
)

# Tono ejecutivo/gerencial, distinto del pedagogico: directo, orientado a impacto
# y decision de negocio, SIN jerga de ML. Usado por el Agente de Reporte Ejecutivo.
EXECUTIVE_SYSTEM_PROMPT = (
    "Eres un consultor de negocio senior que redacta informes para la alta gerencia. "
    "Escribes en espanol, con tono ejecutivo: frases cortas, directas, orientadas a "
    "impacto y a la accion. NO usas jerga tecnica de Machine Learning (nada de "
    "'modelo', 'accuracy', 'hiperparametros', 'clustering'): traduces todo a lenguaje "
    "de negocio. REGLA ABSOLUTA: solo puedes usar las cifras y hechos que se te "
    "entregan explicitamente; jamas inventes numeros, porcentajes ni datos. Si un "
    "dato no esta, no lo menciones. Tus recomendaciones son accionables por una "
    "gerencia (donde enfocar presupuesto, que investigar, que monitorear), nunca "
    "tareas tecnicas."
)


def _get_client():
    """Devuelve un cliente Groq o None si no es posible usarlo."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq

        return Groq(api_key=api_key)
    except Exception:
        return None


def is_llm_available() -> bool:
    return _get_client() is not None


def _domain_preamble(domain: str | None) -> str:
    """Preambulo opcional que pide reformular con vocabulario de dominio.

    Puramente lingüistico: nunca introduce cifras ni datos nuevos, solo indica
    que palabras usar al explicar hechos que YA vienen en el prompt.
    """
    if not domain or domain == "general":
        return ""
    from .domain_config import get_domain

    cfg = get_domain(domain)
    return (
        f"Contexto de negocio: estas trabajando en el dominio de {cfg['label']} "
        f"({cfg['tagline']}). Cuando sea natural, usa vocabulario de este dominio "
        f"(ej. {', '.join(list(cfg['vocab'].values())[:3])}) para explicar los "
        f"mismos hechos, SIN inventar columnas, cifras ni datos nuevos. "
    )


def narrate_stream(
    prompt: str,
    fallback_text: str | None = None,
    domain: str | None = None,
) -> Iterator[str]:
    """Genera tokens de narracion pedagogica.

    `prompt` es la instruccion concreta para el LLM (que explicar).
    `fallback_text` es el texto usado si no hay LLM disponible; si no se pasa,
    se usa el propio prompt como texto base.
    `domain` (opcional) agrega un preambulo de vocabulario de negocio, sin
    alterar los hechos ni las cifras del texto.
    """
    client = _get_client()
    if client is None:
        yield from typewriter(fallback_text or prompt)
        return

    try:
        system_content = SYSTEM_PROMPT + _domain_preamble(domain)
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=400,
            stream=True,
        )
        produced = False
        for chunk in completion:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                produced = True
                yield delta
        if not produced and fallback_text:
            yield from typewriter(fallback_text)
    except Exception:
        # Cualquier fallo de red/cuota: caemos al texto local sin romper el flujo.
        yield from typewriter(fallback_text or prompt)


def narrate_text(
    prompt: str,
    fallback_text: str | None = None,
    domain: str | None = None,
) -> str:
    """Version no-streaming: devuelve el texto completo."""
    return "".join(narrate_stream(prompt, fallback_text, domain=domain))


def executive_system_prompt_for(domain: str | None) -> str:
    """Variante del EXECUTIVE_SYSTEM_PROMPT parametrizada por dominio de negocio.

    Agrega el marco de KPIs correcto (ratios/riesgo en Finanzas, SLA/OTIF en
    Logistica, crecimiento/posicionamiento en Estrategia) sin tocar la regla
    absoluta de no inventar cifras, que se mantiene igual en los 4 casos.
    """
    if not domain or domain == "general":
        return EXECUTIVE_SYSTEM_PROMPT
    from .domain_config import get_domain

    cfg = get_domain(domain)
    return (
        EXECUTIVE_SYSTEM_PROMPT
        + f" Estas redactando para el area de {cfg['label']}. Enmarca tus "
        f"recomendaciones usando el vocabulario y los indicadores propios de este "
        f"dominio ({cfg['kpi_context']}), pero SOLO reformulando los hechos que se "
        f"te entregan; nunca inventes un ratio, KPI o cifra que no este en los datos."
    )


def complete_json(
    prompt: str,
    fallback_obj: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Pide al LLM un objeto JSON estructurado; devuelve `fallback_obj` si falla.

    Reutiliza el mismo mecanismo Groq + fallback que la narracion. Ante cualquier
    problema (sin clave, error de red, JSON invalido) devuelve el objeto de respaldo
    determinista, para que la app nunca se rompa.
    """
    client = _get_client()
    if client is None:
        return fallback_obj

    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            return fallback_obj
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else fallback_obj
    except Exception:
        # Sin conexion, cuota agotada o JSON malformado: usamos el respaldo.
        return fallback_obj
