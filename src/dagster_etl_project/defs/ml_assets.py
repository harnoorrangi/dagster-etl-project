from __future__ import annotations
import datetime as dt
import json
import pickle
from typing import List

import polars as pl
from dagster_duckdb import DuckDBResource
from dagster import AssetIn, asset, AssetKey
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

_NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}
def _is_numeric_dtype(dt: pl.datatypes.DataType) -> bool:
    return dt in _NUMERIC_DTYPES

@asset(
    key_prefix=["ml"],
    group_name="ml_pipeline",
    deps=[AssetKey("mart_iris_features")],
    io_manager_key="duckdb_io_manager",
    kinds=["python", "polars", "duckdb"]
)
def feature_table(database: DuckDBResource):
    
    with database.get_connection() as con:
        df = con.sql("SELECT * FROM analytics.mart_iris_features").pl()
    numeric_cols = [
        "sepal_length", "sepal_width", "petal_length", "petal_width",
        "petal_sepal_ratio", "petal_sepal_w_ratio",
    ]
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols]
        + [pl.col(c).std().alias(f"{c}_std") for c in numeric_cols]
    ).to_dicts()[0]
    for c in numeric_cols:
        s = stats.get(f"{c}_std") or 0.0
        stats[f"{c}_std"] = s if s and s != 0.0 else 1.0
    feats = df.with_columns(
        [
            ((pl.col(c) - pl.lit(stats[f"{c}_mean"])) / pl.lit(stats[f"{c}_std"])).alias(f"{c}_z") for c in numeric_cols
        ]
        + [
            (pl.col("petal_length") * pl.col("petal_width")).alias("petal_area"),
            (pl.col("sepal_length") / (pl.col("petal_length") + 1e-6)).alias("sepal_to_petal_len_ratio"),
            (pl.col("sepal_width") ** 2).alias("sepal_width_sq"),
            (pl.col("petal_width") ** 2).alias("petal_width_sq"),
        ]
    )
    return feats

@asset(
    key_prefix=["ml"],
    group_name="ml_pipeline",
    deps=[AssetKey("feature_table")],
    io_manager_key="duckdb_io_manager",
    kinds=["python", "polars", "duckdb"]
)
def trained_model(feature_table):

    df = feature_table
    classes = sorted(df["species"].unique().to_list())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df = df.with_columns(pl.col("species").replace_strict(class_to_idx).alias("label"))
    exclude = {"species", "label"}
    feature_cols = [c for c, dtp in df.schema.items() if c not in exclude and _is_numeric_dtype(dtp)]
    X = df.select(feature_cols).to_numpy()
    y = df.select("label").to_numpy().ravel()
    n = X.shape[0]
    split = max(int(0.8 * n), 1)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test.size else y_train[:0]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)) if y_pred.size else None,
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)) if y_pred.size else None,
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)) if y_pred.size else None,
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)) if y_pred.size else None,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "features": feature_cols,
        "classes": classes,
    }
    payload = {"sk_model": model, "feature_cols": feature_cols, "classes": classes}
    model_blob = pickle.dumps(payload)
    return pl.DataFrame({
        "version": [f"dt_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"],
        "trained_at": [dt.datetime.utcnow()],
        "model": [model_blob],
        "metrics": [json.dumps(metrics)]
    })

@asset(
    key_prefix=["ml"],
    group_name="ml_pipeline",
    deps=[AssetKey("trained_model"), AssetKey("feature_table")],
    io_manager_key="duckdb_io_manager",
    kinds=["python", "polars", "duckdb"]
)
def predictions(trained_model, feature_table):
    if trained_model.is_empty():
        raise RuntimeError("trained_model is empty; cannot generate predictions")

    latest = trained_model.sort("trained_at", descending=True).row(0, named=True)


    version = latest["version"]
    model_blob = latest["model"]
    if isinstance(model_blob, memoryview):
        model_blob = model_blob.tobytes()
    elif not isinstance(model_blob, (bytes, bytearray)):
        model_blob = bytes(model_blob)

    payload = pickle.loads(model_blob)
    model = payload["sk_model"]                    
    feature_cols: List[str] = payload["feature_cols"]
    classes: List[str] = payload["classes"]

    # Validate columns exist
    missing = [c for c in feature_cols if c not in feature_table.columns]
    if missing:
        raise RuntimeError(f"feature_table missing columns required by model: {missing}")


    X = feature_table.select(feature_cols).to_numpy()

    if X.size == 0:
        return pl.DataFrame(
            schema={
                "predicted_label": pl.Utf8,
                "version": pl.Int64,
                "scored_at": pl.Datetime(time_unit="us", time_zone=None),
            }
        )

    raw_pred = model.predict(X)


    if np.issubdtype(np.array(raw_pred).dtype, np.integer):
        pred_labels = [classes[i] for i in raw_pred]
    else:
        pred_labels = list(map(str, raw_pred))

    out = pl.DataFrame(
        {
            "predicted_label": pred_labels,
            "version": [version] * len(pred_labels),
            "scored_at": [dt.datetime.utcnow()] * len(pred_labels),
        }
    )

    maybe_keys = [c for c in feature_table.columns if c.lower() in {"id", "row_id"}]
    if maybe_keys:
        out = pl.concat([feature_table.select(maybe_keys), out], how="horizontal")

    return out