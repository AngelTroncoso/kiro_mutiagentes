"""Utilidades compartidas por los agentes de ML (capa 3).

Centraliza la preparacion de features (codificacion + escalado) para que los tres
agentes trabajen sobre una matriz numerica coherente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureMatrix:
    X: np.ndarray
    feature_names: list[str]


def build_features(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    max_ohe_cardinality: int = 15,
) -> FeatureMatrix:
    """Convierte un DataFrame en una matriz numerica lista para sklearn.

    - Numericas: se usan tal cual (imputando cualquier NaN residual con la media).
    - Categoricas de baja cardinalidad: one-hot encoding.
    - Categoricas de alta cardinalidad y texto: se omiten (poco utiles sin NLP).
    - Fechas: se expanden en anio, mes y dia.
    """
    exclude = set(exclude or [])
    frames: list[pd.DataFrame] = []
    names: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            frames.append(s.astype(int).to_frame(col))
            names.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            filled = s.fillna(s.mean())
            frames.append(filled.to_frame(col))
            names.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
            frames.append(pd.DataFrame({
                f"{col}_anio": dt.dt.year.fillna(0),
                f"{col}_mes": dt.dt.month.fillna(0),
                f"{col}_dia": dt.dt.day.fillna(0),
            }))
            names.extend([f"{col}_anio", f"{col}_mes", f"{col}_dia"])
        else:
            nun = s.nunique(dropna=True)
            if 1 < nun <= max_ohe_cardinality:
                dummies = pd.get_dummies(s.astype(str), prefix=col)
                frames.append(dummies.astype(int))
                names.extend(list(dummies.columns))
            # alta cardinalidad / texto: se omite

    if not frames:
        # Fallback: matriz vacia con una constante para no romper.
        return FeatureMatrix(X=np.zeros((len(df), 1)), feature_names=["(sin_features)"])

    X = pd.concat(frames, axis=1).to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureMatrix(X=X, feature_names=names)


def encode_target(y: pd.Series) -> tuple[np.ndarray, list[str] | None]:
    """Codifica el objetivo. Devuelve (y_codificado, etiquetas o None si continuo)."""
    if pd.api.types.is_numeric_dtype(y) and not _is_discrete_numeric(y):
        return y.fillna(y.mean()).to_numpy(dtype=float), None
    cats = y.astype("category")
    labels = [str(c) for c in cats.cat.categories]
    codes = cats.cat.codes.to_numpy()
    # Reemplazar -1 (NaN) por la clase mayoritaria.
    if (codes == -1).any():
        majority = np.bincount(codes[codes >= 0]).argmax() if (codes >= 0).any() else 0
        codes = np.where(codes == -1, majority, codes)
    return codes, labels


def _is_discrete_numeric(y: pd.Series) -> bool:
    nun = y.nunique(dropna=True)
    return nun <= max(10, int(0.05 * len(y))) and nun <= 20
