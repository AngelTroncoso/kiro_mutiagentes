# 🤖 Analista de Datos Multiagente

Aplicación web con interfaz en **streaming** donde un equipo de agentes de IA
limpia un dataset, lo diagnostica, ejecuta Machine Learning en sus tres paradigmas
(supervisado, no supervisado y por refuerzo) y recomienda el mejor enfoque según tu
objetivo. Cada paso se narra **en tiempo real y en lenguaje simple**, así que sirve
tanto para trabajo profesional como para aprender.

---

## Arquitectura: 5 capas

| Capa | Agente | Rol |
|------|--------|-----|
| 0 | **Orquestador** (`orchestrator/graph.py`) | Controla el pipeline con LangGraph y transmite eventos a la UI. |
| 1 | **ETL** (`agents/etl_agent.py`) | Detecta tipos, nulos, duplicados y outliers; explica y aplica la limpieza. |
| 2 | **Router** (`agents/router_agent.py`) | Diagnostica qué ML aplica y por qué; pre-marca el selector. |
| 3 | **Supervisado** (`agents/supervised_agent.py`) | Clasificación o regresión con varios modelos. |
| 3 | **No supervisado** (`agents/unsupervised_agent.py`) | K-Means + PCA para descubrir grupos. |
| 3 | **Refuerzo** (`agents/reinforcement_agent.py`) | Contextual bandit (LinUCB) sobre un entorno simulado. |
| 4 | **Recomendador** (`agents/recommender_agent.py`) | Compara enfoques de forma normalizada y recomienda. |
| 6 | **Reporte Ejecutivo** (`agents/executive_report_agent.py`) | Traduce todo a un informe gerencial accionable y descargable. |

El pipeline corre en **dos fases** porque hay una decisión humana en el medio:

```
Fase 1:  ETL ──> Router ──> [ el usuario elige qué analizar ]
Fase 2:  dispatch ──(condicional)──> {supervisado, no supervisado, refuerzo} ──> recomendador ──> reporte ejecutivo
```

La **capa 6** cierra el pipeline: toma el diagnóstico, los resultados de los tres
enfoques y la recomendación, y genera un **informe gerencial** con tono ejecutivo
(título, resumen, hallazgo clave, 3 recomendaciones accionables y siguiente paso).
Nunca inventa cifras: todo número es trazable a `clean_df`, `diagnosis` o las
métricas calculadas. Si no hay LLM, arma el reporte con plantillas deterministas
sobre los datos reales. Es **descargable en Markdown** desde la interfaz.

La ramificación condicional de la Fase 2 se modela con `add_conditional_edges` de
LangGraph: solo se ejecutan los enfoques que el usuario dejó seleccionados.

---

## Stack técnico

- **Orquestación**: LangGraph (grafo de estados con ramificación condicional)
- **Interfaz**: Streamlit + `st.write_stream` para streaming token a token
- **ML**: scikit-learn (supervisado/no supervisado), LinUCB propio + Gymnasium (refuerzo)
- **Datos**: pandas, numpy, openpyxl
- **Visualización**: Plotly (interactivo)
- **LLM**: Groq (rápido, ideal para streaming) con **fallback local** si no hay clave

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

### Clave de Groq (opcional pero recomendado)

La narración pedagógica usa Groq. Si defines la clave, verás explicaciones
generadas por IA en streaming real; si no, la app usa un **narrador local**
determinista (igual de pedagógico, sin coste ni conexión).

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "tu_clave_aqui"
# opcional: $env:GROQ_MODEL = "llama-3.3-70b-versatile"
```

```bash
# Linux/Mac
export GROQ_API_KEY="tu_clave_aqui"
```

---

## Uso

```bash
streamlit run app.py
```

Luego, en el navegador:

1. **Sube** un archivo `.csv` o `.xlsx`.
2. Pulsa **Iniciar diagnóstico**: el ETL limpia y narra; el Router diagnostica.
3. Ajusta la **variable objetivo**, los **enfoques a ejecutar** (pre-marcados) y tu
   **objetivo de negocio** (predecir / segmentar / decidir / explorar).
4. Pulsa **Ejecutar análisis**: cada agente entrena, grafica y explica en vivo.
5. Lee la **recomendación final** destacada, condicionada a tu objetivo.
6. Revisa el **reporte ejecutivo** al cierre y descárgalo en Markdown para compartirlo.

---

## Prueba rápida sin interfaz

`smoke_test.py` corre el pipeline completo headless sobre datasets sintéticos
(clasificación y regresión) y verifica cada capa:

```bash
python smoke_test.py
```

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

---

## Estructura del proyecto

```
analista_datos_multiagente/
├── app.py                      # entrada Streamlit (UI + streaming)
├── smoke_test.py               # prueba headless del pipeline
├── orchestrator/
│   ├── graph.py                # pipeline LangGraph (2 fases, ramificación condicional)
│   └── state.py                # estado compartido (PipelineState)
├── agents/
│   ├── common.py               # preparación de features compartida
│   ├── etl_agent.py            # Capa 1
│   ├── router_agent.py         # Capa 2
│   ├── supervised_agent.py     # Capa 3
│   ├── unsupervised_agent.py   # Capa 3
│   ├── reinforcement_agent.py  # Capa 3
│   ├── recommender_agent.py    # Capa 4
│   └── executive_report_agent.py  # Capa 6 (reporte gerencial)
├── utils/
│   ├── data_loading.py         # carga y perfilado de datasets
│   ├── streaming.py            # EventBus + puente hilo↔UI
│   ├── llm.py                  # narración Groq + fallback local
│   └── visualization.py        # fábricas de gráficos Plotly
├── requirements.txt
└── README.md
```
