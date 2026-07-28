"""Carga y perfilado inicial de datasets (.csv / .xlsx)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype_kind: str  # "numeric" | "categorical" | "datetime" | "text" | "boolean"
    n_missing: int
    pct_missing: float
    n_unique: int
    sample_values: list[Any] = field(default_factory=list)


def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Lee un CSV o Excel desde bytes en un DataFrame.

    Para CSV intenta autodetectar separador y codificacion comunes.
    """
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    # CSV: probamos combinaciones habituales de encoding y separador.
    last_err: Exception | None = None
    for encoding in ("utf-8", "latin-1"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    sep=sep,
                    encoding=encoding,
                    engine="python",
                )
                if df.shape[1] >= 1 and not (df.shape[1] == 1 and sep != ","):
                    return df
            except Exception as exc:  # noqa: BLE001
                last_err = exc
    # Ultimo intento con inferencia total.
    try:
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No se pudo leer el archivo como CSV o Excel: {last_err or exc}")


def _classify_column(series: pd.Series) -> str:
    """Clasifica una columna en un tipo semantico simple."""
    non_null = series.dropna()
    if non_null.empty:
        return "text"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Intento de parseo a fecha para columnas objeto.
    sample = non_null.astype(str).head(50)
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            return "datetime"
    except Exception:  # noqa: BLE001
        pass

    n_unique = non_null.nunique()
    n_total = len(non_null)
    avg_len = non_null.astype(str).str.len().mean()
    # Muchos valores unicos y textos largos -> texto libre; si no, categorico.
    if n_unique > 50 and (n_unique / n_total) > 0.5 and avg_len > 20:
        return "text"
    return "categorical"


def profile_dataframe(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        kind = _classify_column(series)
        sample = series.dropna().unique()[:5].tolist()
        profiles.append(
            ColumnProfile(
                name=str(col),
                dtype_kind=kind,
                n_missing=n_missing,
                pct_missing=round(100.0 * n_missing / n_rows, 2) if n_rows else 0.0,
                n_unique=int(series.nunique()),
                sample_values=[_json_safe(v) for v in sample],
            )
        )
    return profiles


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def columns_by_kind(profiles: list[ColumnProfile]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": [],
    }
    for p in profiles:
        out.setdefault(p.dtype_kind, []).append(p.name)
    return out


def basic_overview(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_duplicates": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }
