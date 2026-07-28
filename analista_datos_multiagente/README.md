# 🤖 Analista de Datos Multiagente

Aplicación web con interfaz en **streaming** donde un equipo de agentes de IA
limpia un dataset, lo diagnostica, ejecuta Machine Learning en sus tres paradigmas
(supervisado, no supervisado y por refuerzo) y recomienda el mejor enfoque según tu
objetivo. Cada paso se narra **en tiempo real y en lenguaje simple**, así que sirve
tanto para trabajo profesional como para aprender.

Sobre el pipeline base se agregó una **capa de dominio de negocio** (Finanzas /
Logística / Estrategia) que reconfigura vocabulario y sugerencias sin tocar nunca
la matemática, un **rediseño visual premium**, y **5 módulos de innovación**
(voz, simulación financiera, mapas logísticos, escenarios estratégicos y
video-resumen) que se activan según el dominio elegido.

---

## Arquitectura: 11 capas

| Capa | Módulo | Rol |
|------|--------|-----|
| 0 | **Dominio de negocio** (`utils/domain_config.py`, `utils/theme.py`) | Selector Finanzas/Logística/Estrategia: reconfigura vocabulario, sugerencias y acento visual. Nunca cambia una cifra. |
| 0 | **Orquestador** (`orchestrator/graph.py`) | Controla el pipeline con LangGraph y transmite eventos a la UI. |
| 1 | **ETL** (`agents/etl_agent.py`) | Detecta tipos, nulos, duplicados y outliers; explica y aplica la limpieza con el vocabulario del dominio activo. |
| 2 | **Router** (`agents/router_agent.py`) | Diagnostica qué ML aplica y por qué; enmarca la variable objetivo según el dominio; pre-marca el selector. |
| 3 | **Supervisado** (`agents/supervised_agent.py`) | Clasificación o regresión con varios modelos; conserva el modelo entrenado para reutilizarlo en la Capa 10. |
| 3 | **No supervisado** (`agents/unsupervised_agent.py`) | K-Means + PCA para descubrir grupos; conserva el perfil de clusters para la Capa 10. |
| 3 | **Refuerzo** (`agents/reinforcement_agent.py`) | Contextual bandit (LinUCB) sobre un entorno simulado. |
| 4 | **Recomendador** (`agents/recommender_agent.py`) | Compara enfoques de forma normalizada; el listado de objetivos de negocio se curva por dominio. |
| 6 | **Reporte Ejecutivo** (`agents/executive_report_agent.py`) | Traduce todo a un informe gerencial accionable, con KPIs propios del dominio, y descargable en Markdown. |
| 7 | **Chat de voz** (`utils/voice.py`, `agents/voice_qa_agent.py`) | Preguntas en lenguaje natural sobre lo ya calculado, con voz de entrada/salida. |
| 8 | **Simulador financiero Monte Carlo** (`utils/finance_sim.py`) | Proyección de flujo de caja con incertidumbre. Solo Finanzas. |
| 9 | **Rutas e inventario animado** (`utils/logistics_viz.py`) | Mapa de rutas y evolución de inventario vs. punto de reorden. Solo Logística. |
| 10 | **War Room de escenarios** (`utils/war_room.py`) | Sliders what-if sobre el modelo ya entrenado + matriz BCG sobre los clusters ya calculados. Solo Estrategia. |

El pipeline principal (capas 1-6) corre en **dos fases** porque hay una decisión
humana en el medio:

```
Fase 1:  ETL ──> Router ──> [ el usuario elige qué analizar ]
Fase 2:  dispatch ──(condicional)──> {supervisado, no supervisado, refuerzo} ──> recomendador ──> reporte ejecutivo
```

La ramificación condicional de la Fase 2 se modela con `add_conditional_edges` de
LangGraph: solo se ejecutan los enfoques que el usuario dejó seleccionados. Los
módulos de innovación (capas 7-10) viven en una sección aparte de la interfaz
("Innovación") y se apoyan en los resultados que el pipeline principal ya calculó;
ninguno reentrana modelos ni recalcula métricas.

---

## Capa 0 — Dominio de negocio

Antes de cargar el dataset, eliges **Finanzas**, **Logística** o **Estrategia**
(o ningún dominio, modo "General"). Esa elección es una capa de **presentación**,
nunca de cómputo: cambiar de dominio sobre el mismo dataset produce exactamente
las mismas métricas (accuracy, silhouette, score_norm), verificado en el smoke
test. Lo que cambia:

- **ETL/Router**: el vocabulario con el que se explica lo ya detectado (ej. una
  columna de churn se enmarca como "riesgo de fuga de ingresos" en Finanzas, o
  "indicador de cumplimiento de entrega" en Logística).
- **Recomendador**: la lista de objetivos de negocio ofrecida (`business_goal`)
  usa etiquetas curadas por dominio, sobre las mismas 4 claves internas
  (predecir/segmentar/decidir/explorar) y los mismos pesos matemáticos.
- **Reporte Ejecutivo**: el marco de KPIs (ratios de riesgo en Finanzas, SLA/OTIF
  en Logística, crecimiento/posicionamiento en Estrategia).
- **Interfaz**: acento de color (esmeralda/dorado en Finanzas, cian/naranjo en
  Logística, violeta/magenta en Estrategia), tema oscuro tipo dashboard premium
  con tipografía Space Grotesk/Inter, stepper de fases y tarjetas "glass".
- **Innovación**: las capas 8, 9 y 10 solo se muestran si el dominio activo
  coincide (con aviso si estás en modo General).

Cada dataset de ejemplo (`datos_ejemplo_finanzas.csv`, `datos_ejemplo_logistica.csv`,
`datos_ejemplo_estrategia.csv`) está pensado para probar su dominio correspondiente.

---

## Capa 6 — Reporte Ejecutivo

Cierra el pipeline principal: toma el diagnóstico, los resultados de los tres
enfoques y la recomendación, y genera un **informe gerencial** con tono ejecutivo
(título, resumen, hallazgo clave, 3 recomendaciones accionables y siguiente paso).
Nunca inventa cifras: todo número es trazable a `clean_df`, `diagnosis` o las
métricas calculadas. Si no hay LLM, arma el reporte con plantillas deterministas
sobre los datos reales. Es **descargable en Markdown** desde la interfaz.

---

## Capa 7 — Chat de voz con el Analista

Sección "🎙️ Innovación" donde preguntas en lenguaje natural ("¿cuál fue el mejor
modelo?", "¿qué variable importa más?") y recibes respuesta en texto y, si activas
el modo manos libres, también en audio.

- **Entrada de voz**: `st.audio_input` nativo de Streamlit (pide permiso de
  micrófono de forma confiable); si el entorno tiene una versión vieja de
  Streamlit, cae a `audio-recorder-streamlit` como respaldo; si nada de eso
  funciona, queda el campo de texto.
- **STT**: Groq Whisper API (`whisper-large-v3`, fijado en español), misma
  `GROQ_API_KEY` del resto de la app.
- **TTS**: ElevenLabs (`ELEVENLABS_API_KEY`) con fallback a gTTS (sin clave,
  requiere internet) y fallback final a solo texto.
- **Anti-invención**: el agente (`agents/voice_qa_agent.py`) solo puede citar
  hechos ya calculados por los agentes anteriores, siempre menciona la fuente
  ("según el Agente Supervisado...") y valida automáticamente que ningún número
  de la respuesta del LLM sea ajeno a esos hechos; si lo es, descarta la
  respuesta y usa una plantilla determinista.

---

## Capa 8 — Simulador Monte Carlo de flujo de caja (Finanzas)

Proyecta flujo de caja con incertidumbre a partir de una columna monetaria real
del dataset:

- Los parámetros de la simulación (media y desviación estándar de entradas y
  salidas) se derivan de la columna elegida, nunca se inventan; si el dataset
  mezcla signos, se separan automáticamente en entradas/salidas.
- Motor Monte Carlo (`utils/finance_sim.py`, numpy, 5 000-10 000 iteraciones):
  VAN/NPV, TIR/IRR (vía `scipy.optimize.brentq` sobre el flujo promedio), y los
  percentiles P10/P50/P90 del flujo acumulado, más la probabilidad de flujo
  negativo.
- Si el enfoque ganador de la Capa 4 fue supervisado sobre una variable binaria
  (ej. churn), se puede ajustar la simulación con la tasa real detectada por ese
  agente, en vez de una probabilidad inventada por fila.
- Visualización: banda de confianza (*fan chart*) e histograma del VAN, con la
  plantilla oscura de `theme.py`.

Verificado en el smoke test contra una fórmula cerrada (caso determinista, sin
varianza): el NPV simulado coincide con el NPV analítico y la IRR calculada
anula el NPV.

---

## Capa 9 — Rutas e inventario animado (Logística)

Dos visualizaciones independientes, cada una con su propia degradación segura:

- **Mapa de rutas** (pydeck `ArcLayer`): si el dataset tiene columnas de
  latitud/longitud reales, se grafican directo; si solo tiene una columna
  categórica de ubicación (ciudad/ruta), la interfaz pide un mapeo manual
  explícito de coordenadas por valor — nunca se inventa una coordenada por
  defecto. Las rutas con retraso se colorean distinto (según la columna de SLA
  detectada).
- **Animación de inventario** (Plotly *frames*): evolución del stock vs. un
  punto de reorden calculado como el percentil 20 real de la propia columna de
  stock (método simple y declarado explícitamente, no una fórmula de demanda ×
  lead time que el dataset no tiene datos para sustentar).
- Si no hay ninguna columna usable, el módulo lo dice claramente en vez de
  mostrar una animación vacía o inventada.

---

## Capa 10 — War Room de escenarios estratégicos (Estrategia)

Panel *what-if* que reutiliza artefactos ya calculados, sin reentrenar nada:

- **Sliders sobre el modelo ya entrenado**: parte de la fila "promedio real"
  del dataset (mediana/moda por columna) y, al mover un slider, reconstruye el
  vector de features en el mismo orden de entrenamiento para re-predecir con
  el modelo y el *scaler* que ya ajustó el Agente Supervisado (`trained_model`,
  `trained_scaler` guardados en `supervised_result`).
- **Matriz BCG**: ubica los clusters ya calculados por el Agente No Supervisado
  en los cuadrantes Estrella/Vaca lechera/Interrogante/Perro, comparando cada
  cluster contra la media global real de dos columnas numéricas elegibles (no
  contra un umbral inventado). El resumen ejecutivo de la matriz lo redacta el
  LLM reformulando esos números, nunca generando cifras nuevas.

---

## Stack técnico

- **Orquestación**: LangGraph (grafo de estados con ramificación condicional)
- **Interfaz**: Streamlit + `st.write_stream` para streaming token a token, tema
  oscuro custom (`utils/theme.py`) y panel de resultados reordenable
  (`streamlit-sortables`, con degradación a layout fijo si no está disponible)
- **ML**: scikit-learn (supervisado/no supervisado), LinUCB propio + Gymnasium (refuerzo)
- **Datos**: pandas, numpy, openpyxl
- **Visualización**: Plotly (interactivo), pydeck (mapas de la Capa 9)
- **Simulación**: numpy + scipy (Monte Carlo financiero de la Capa 8)
- **Voz**: Groq Whisper (STT), ElevenLabs con fallback gTTS (TTS)
- **LLM**: Groq (rápido, ideal para streaming) con **fallback local** determinista
  si no hay clave, en todos los módulos (narración, reporte ejecutivo, chat de
  voz, resumen BCG)

---

## Instalación

```bash
# 1. (opcional) crea un entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. instala dependencias
pip install -r requirements.txt
```

### Claves de API (opcionales, con fallback si no las defines)

| Variable | Para qué | Si no está |
|----------|----------|------------|
| `GROQ_API_KEY` | Narración por IA (todas las capas) y transcripción de voz (Capa 7) | Narrador local determinista; transcripción deshabilitada (usa el campo de texto) |
| `ELEVENLABS_API_KEY` | Síntesis de voz de alta calidad (Capa 7) | Fallback a gTTS; si tampoco hay internet, solo texto |

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "tu_clave_aqui"
$env:ELEVENLABS_API_KEY = "tu_clave_aqui"   # opcional
# opcional: $env:GROQ_MODEL = "llama-3.3-70b-versatile"
```

```bash
# Linux/Mac
export GROQ_API_KEY="tu_clave_aqui"
export ELEVENLABS_API_KEY="tu_clave_aqui"
```

**Nunca subas estas claves a un archivo del repo.** Si las compartes en texto
plano en algún momento, rótalas después.

---

## Uso

```bash
streamlit run app.py
```

Luego, en el navegador:

1. **Elige un dominio de negocio** (Finanzas / Logística / Estrategia), o
   continúa en modo General.
2. **Sube** un archivo `.csv` o `.xlsx` (o usa el dataset de ejemplo de tu dominio).
3. Pulsa **Iniciar diagnóstico**: el ETL limpia y narra; el Router diagnostica,
   ya con el vocabulario del dominio elegido.
4. Ajusta la **variable objetivo**, los **enfoques a ejecutar** (pre-marcados) y tu
   **objetivo de negocio** (etiquetas curadas por dominio).
5. Pulsa **Ejecutar análisis**: cada agente entrena, grafica y explica en vivo, en
   un panel de tarjetas reordenable.
6. Lee la **recomendación final** destacada, condicionada a tu objetivo.
7. Revisa el **reporte ejecutivo** al cierre y descárgalo en Markdown para compartirlo.
8. Explora la sección **🎙️ Innovación**: pregúntale al Analista por voz o texto,
   y si tu dominio es Finanzas/Logística/Estrategia, usa el módulo específico
   (Monte Carlo, mapa de rutas, o War Room).

---

## Prueba rápida sin interfaz

`smoke_test.py` corre el pipeline completo headless sobre datasets sintéticos
(clasificación y regresión) cruzados con los 3 dominios de negocio, más los
módulos de innovación:

```bash
python smoke_test.py
```

Verifica, entre otras cosas:

- El pipeline completa sin error en cada combinación de caso × dominio.
- Las métricas (accuracy, silhouette, score_norm) son **idénticas** entre
  dominios para el mismo dataset (el dominio es solo presentación).
- El chat de voz (Capa 7) responde 5 preguntas mockeadas citando siempre la
  fuente y sin ninguna cifra ausente del estado ya calculado.
- El motor Monte Carlo (Capa 8) es matemáticamente consistente con una fórmula
  cerrada, y sus parámetros son 100% trazables a columnas reales.
- El módulo logístico (Capa 9) no inventa coordenadas cuando no las hay, y
  genera el mapa/animación sin excepciones cuando sí hay datos suficientes.

Debe terminar con `RESULTADO GLOBAL: OK`.

---

## Nota honesta sobre el aprendizaje por refuerzo

El refuerzo clásico necesita un entorno interactivo con estados, acciones y
recompensas. Un dataset tabular estático no lo es. Por eso el Agente por Refuerzo
construye un **entorno simulado**: cada fila es un contexto, las acciones son los
valores de la variable objetivo, y la recompensa es 1 si el agente acierta. Esto es
un **contextual bandit** (implementado con LinUCB), la aproximación honesta y
explicable de RL sobre datos tabulares. El agente lo declara explícitamente en su
narración.

## Nota honesta sobre los módulos de innovación

Las capas 7 a 10 siguen el mismo principio que el resto de la app: ningún número
que se muestra en pantalla se inventa. El chat de voz solo cita hechos ya
calculados; el simulador financiero deriva sus distribuciones de columnas reales;
el mapa logístico no dibuja coordenadas que no existan; y el War Room reutiliza el
modelo y los clusters ya entrenados en vez de generar predicciones nuevas sin base.
Cuando falta información suficiente para alguno de estos módulos, la interfaz lo
dice explícitamente en vez de rellenar con datos sintéticos.

---

## Estructura del proyecto

```
analista_datos_multiagente/
├── app.py                         # entrada Streamlit (UI + streaming + Innovación)
├── smoke_test.py                  # prueba headless: pipeline x dominios + innovación
├── orchestrator/
│   ├── graph.py                   # pipeline LangGraph (2 fases, ramificación condicional)
│   └── state.py                   # estado compartido (PipelineState, incluye domain)
├── agents/
│   ├── common.py                  # preparación de features compartida
│   ├── etl_agent.py               # Capa 1
│   ├── router_agent.py            # Capa 2
│   ├── supervised_agent.py        # Capa 3 (conserva el modelo entrenado)
│   ├── unsupervised_agent.py      # Capa 3 (conserva el perfil de clusters)
│   ├── reinforcement_agent.py     # Capa 3
│   ├── recommender_agent.py       # Capa 4
│   ├── executive_report_agent.py  # Capa 6 (reporte gerencial)
│   └── voice_qa_agent.py          # Capa 7 (QA sobre el estado, anti-invención)
├── utils/
│   ├── data_loading.py            # carga y perfilado de datasets
│   ├── streaming.py               # EventBus + puente hilo↔UI
│   ├── llm.py                     # narración Groq + fallback local
│   ├── domain_config.py           # Capa 0: vocabulario/objetivos/color por dominio
│   ├── theme.py                   # Capa 0: CSS premium + plantilla Plotly oscura
│   ├── visualization.py           # fábricas de gráficos Plotly
│   ├── voice.py                   # Capa 7: STT (Groq Whisper) + TTS (ElevenLabs/gTTS)
│   ├── finance_sim.py             # Capa 8: motor Monte Carlo de flujo de caja
│   ├── logistics_viz.py           # Capa 9: mapa de rutas + inventario animado
│   └── war_room.py                # Capa 10: what-if + matriz BCG
├── datos_ejemplo_clientes.csv     # dataset genérico de ejemplo
├── datos_ejemplo_finanzas.csv     # dataset de ejemplo para el dominio Finanzas
├── datos_ejemplo_logistica.csv    # dataset de ejemplo para el dominio Logística
├── datos_ejemplo_estrategia.csv   # dataset de ejemplo para el dominio Estrategia
├── requirements.txt
└── README.md
```
