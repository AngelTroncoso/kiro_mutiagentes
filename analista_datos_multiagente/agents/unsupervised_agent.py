"""Capa 3 - Agente No Supervisado (clustering + PCA).

- Selecciona k con el metodo del codo + silhouette score.
- Ejecuta K-Means y proyecta con PCA a 2D/3D para visualizar los clusters.
- Explica que agrupaciones emergieron y que las distingue.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import visualization as viz
from utils.llm import narrate_stream

from .common import build_features

AGENT = "NoSupervisado"


def run_unsupervised(state) -> dict[str, Any]:
    bus = state["bus"]
    df: pd.DataFrame = state["clean_df"]
    diagnosis = state.get("diagnosis", {})
    target = state.get("target_column") or diagnosis.get("target_column")

    bus.agent_start(AGENT, "Agente No Supervisado - descubrir grupos", icon="🔍")

    # Excluimos el objetivo del clustering (queremos estructura no supervisada).
    feats = build_features(df, exclude=[target] if target in df.columns else [])
    if feats.X.shape[1] < 2 or feats.X.shape[0] < 10:
        bus.warning(AGENT, "No hay suficientes variables numericas para agrupar.")
        bus.agent_end(AGENT, "Omitido: datos insuficientes.")
        return {"unsupervised_result": {"status": "skipped", "reason": "datos insuficientes"}}

    X = StandardScaler().fit_transform(feats.X)
    n_samples = X.shape[0]

    intro = (
        "Ahora busco grupos naturales en los datos sin usar ninguna etiqueta. Esto se "
        "llama clustering: junta filas parecidas entre si. Primero pruebo distintos "
        "numeros de grupos (k) y elijo el que mejor separa los datos."
    )
    bus.narrate(AGENT, narrate_stream(
        "Explica a un principiante que es el clustering y por que hay que elegir el "
        f"numero de grupos. Contexto: {intro}", intro))

    # --- Seleccion de k ---
    k_max = int(min(8, max(2, n_samples // 5)))
    ks = list(range(2, k_max + 1)) or [2]
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        try:
            silhouettes.append(float(silhouette_score(X, lbl)))
        except Exception:  # noqa: BLE001
            silhouettes.append(0.0)

    best_idx = int(np.argmax(silhouettes))
    best_k = ks[best_idx]
    best_sil = silhouettes[best_idx]

    bus.chart(AGENT, "Seleccion de k", viz.elbow_plot(ks, inertias, silhouettes),
              caption="La inercia baja siempre; el silhouette (0 a 1) marca el mejor equilibrio.")

    # --- Clustering final + PCA ---
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    n_pca = min(3, X.shape[1])
    coords = PCA(n_components=n_pca, random_state=42).fit_transform(X)
    bus.chart(AGENT, "Clusters (PCA)",
              viz.cluster_scatter(coords, labels, dims=n_pca),
              caption="Cada color es un grupo. PCA resume muchas columnas en 2-3 ejes.")

    # Tamano de cada cluster.
    sizes = pd.Series(labels).value_counts().sort_index()
    size_df = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in sizes.index],
        "Filas": sizes.values,
        "% del total": (sizes.values / n_samples * 100).round(1),
    })
    bus.table(AGENT, "Tamano de cada grupo", size_df)

    # Perfil: media de las numericas por cluster (para explicar diferencias).
    numeric_cols = [c for c in diagnosis.get("columns", {}).get("numeric", [])
                    if c in df.columns and c != target]
    profile_txt = ""
    if numeric_cols:
        prof = df[numeric_cols].copy()
        prof["cluster"] = labels
        means = prof.groupby("cluster").mean(numeric_only=True)
        distinctive = []
        for c in labels_unique(labels):
            row = means.loc[c]
            top_col = (row - means.mean()).abs().idxmax()
            direction = "alto" if row[top_col] > means[top_col].mean() else "bajo"
            distinctive.append(f"Cluster {c}: {top_col} {direction}")
        profile_txt = "; ".join(distinctive)

    quality = ("buena" if best_sil > 0.5 else "moderada" if best_sil > 0.25 else "debil")
    fallback = (
        f"Encontre {best_k} grupos naturales con una separacion {quality} "
        f"(silhouette {best_sil:.2f}, donde 1 es ideal). "
        + (f"Lo que mas distingue a los grupos: {profile_txt}. " if profile_txt else "")
        + "El silhouette mide que tan bien separado esta cada grupo del resto."
    )
    bus.narrate(AGENT, narrate_stream(
        "Explica a un principiante que grupos apareceron y que significa el silhouette. "
        f"Resultado: {fallback}", fallback))

    result = {
        "status": "ok",
        "best_k": best_k,
        "silhouette": best_sil,
        "cluster_sizes": {int(i): int(v) for i, v in sizes.items()},
        "quality": quality,
        "distinctive": profile_txt,
        "score_norm": float(np.clip((best_sil + 1) / 2, 0, 1)),
    }
    bus.result(AGENT, "unsupervised", result)
    bus.agent_end(AGENT, f"{best_k} clusters (silhouette={best_sil:.2f}).")
    return {"unsupervised_result": result}


def labels_unique(labels: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(labels))
