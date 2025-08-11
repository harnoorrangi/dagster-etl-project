# src/dagster_etl_project/defs/ml_assets.py

from __future__ import annotations

import datetime as dt
import json
import pickle
from typing import Dict, List

import polars as pl
from dagster import AssetKey, MetadataValue, asset
from dagster_duckdb import DuckDBResource
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


# ---- helpers -----------------------------------------------------------------

_NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}
def _is_numeric_dtype(dt: pl.datatypes.DataType) -> bool:
    # works across polars versions without relying on private APIs
    return dt in _NUMERIC_DTYPES


# 1) Feature engineering (reads dbt mart table, writes ml table)
@asset(
    key=AssetKey(["ml", "iris_features"]),
    group_name="ml",
    deps={AssetKey(["mart_iris_features"])},
    kinds=("python", "polars", "duckdb") # depend on dbt model asset
)
def iris_features(context, database: DuckDBResource) -> None:
    with database.get_connection() as con:
        df = con.sql("SELECT * FROM analytics.mart_iris_features").pl()

        numeric_cols: List[str] = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
            "petal_sepal_ratio", "petal_sepal_w_ratio",
        ]

        stats: Dict[str, float] = df.select(
            [pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols]
            + [pl.col(c).std().alias(f"{c}_std") for c in numeric_cols]
        ).to_dicts()[0]

        for c in numeric_cols:
            s = stats.get(f"{c}_std") or 0.0
            stats[f"{c}_std"] = s if s and s != 0.0 else 1.0

        feats = df.with_columns(
            [
                *[
                    ((pl.col(c) - pl.lit(stats[f"{c}_mean"])) / pl.lit(stats[f"{c}_std"])).alias(f"{c}_z")
                    for c in numeric_cols
                ],
                (pl.col("petal_length") * pl.col("petal_width")).alias("petal_area"),
                (pl.col("sepal_length") / (pl.col("petal_length") + 1e-6)).alias("sepal_to_petal_len_ratio"),
                (pl.col("sepal_width") ** 2).alias("sepal_width_sq"),
                (pl.col("petal_width") ** 2).alias("petal_width_sq"),
            ]
        )

        con.execute("CREATE SCHEMA IF NOT EXISTS ml;")
        con.register("fe_df", feats.to_pandas())
        con.execute("CREATE OR REPLACE TABLE ml.iris_features AS SELECT * FROM fe_df;")
        con.unregister("fe_df")

        context.add_output_metadata(
            {
                "row_count": MetadataValue.int(feats.height),
                "columns": MetadataValue.json(sorted(feats.columns)),
                "source_table": "analytics.mart_iris_features",
                "dest_table": "ml.iris_features",
            }
        )


# 2) Train DecisionTree, store blob + metrics in DuckDB
@asset(
    key=AssetKey(["ml", "iris_model"]),
    group_name="ml",
    deps={AssetKey(["ml", "iris_features"])},
    kinds=("python", "polars", "sklearn"),
)
def iris_model(context, database: DuckDBResource) -> None:
    with database.get_connection() as con:
        df = con.sql("SELECT * FROM ml.iris_features").pl()

        if "species" not in df.columns:
            raise ValueError("Expected 'species' column in features for training.")

        classes = sorted(df["species"].unique().to_list())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        df = df.with_columns(pl.col("species").replace_strict(class_to_idx).alias("label"))

        exclude = {"species", "label"}
        feature_cols = [
            c for c, dtp in df.schema.items()
            if c not in exclude and _is_numeric_dtype(dtp)
        ]

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

        version = f"dt_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        trained_at = dt.datetime.utcnow()
        metrics_json = json.dumps(metrics)

        con.execute("CREATE SCHEMA IF NOT EXISTS ml;")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ml.iris_model (
                version TEXT,
                trained_at TIMESTAMP,
                model BLOB,
                metrics JSON
            );
            """
        )
        con.execute(
            "INSERT INTO ml.iris_model (version, trained_at, model, metrics) VALUES (?, ?, ?, ?);",
            [version, trained_at, model_blob, metrics_json],
        )

        context.add_output_metadata(
            {
                "version": version,
                "trained_at": trained_at.isoformat() + "Z",
                "metrics": MetadataValue.json(metrics),
                "model_table": "ml.iris_model",
            }
        )


# 3) Inference — load latest model and score all rows, persist predictions
@asset(
    key=AssetKey(["ml", "iris_predictions"]),
    group_name="ml",
    deps={AssetKey(["ml", "iris_model"]), AssetKey(["ml", "iris_features"])},
    kinds=("python", "polars", "sklearn")
)
def iris_inference(context, database: DuckDBResource) -> None:
    with database.get_connection() as con:
        latest = con.sql(
            """
            SELECT version, trained_at, model, metrics
            FROM ml.iris_model
            ORDER BY trained_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not latest:
            raise RuntimeError("No rows found in ml.iris_model; train the model first.")

        version, trained_at, model_blob, metrics_json = latest
        payload = pickle.loads(model_blob)
        model: DecisionTreeClassifier = payload["sk_model"]
        feature_cols: List[str] = payload["feature_cols"]
        classes: List[str] = payload["classes"]

        feats = con.sql("SELECT * FROM ml.iris_features").pl()
        X = feats.select(feature_cols).to_numpy()
        y_hat_idx = model.predict(X) if X.size else []
        pred_labels = [classes[i] for i in y_hat_idx]

        pred_df = pl.DataFrame(
            {
                "predicted_label": pred_labels,
                "version": [version] * len(pred_labels),
                "scored_at": [dt.datetime.utcnow()] * len(pred_labels),
            }
        )

        maybe_keys = [c for c in feats.columns if c.lower() in {"id", "row_id"}]
        if maybe_keys:
            pred_df = pl.concat([feats.select(maybe_keys), pred_df], how="horizontal")

        eval_meta = {}
        if "species" in feats.columns and len(pred_labels) == feats.height:
            truth = feats.get_column("species").to_list()
            acc = accuracy_score(truth, pred_labels)
            f1m = f1_score(truth, pred_labels, average="macro", zero_division=0)
            eval_meta = {"eval_accuracy": acc, "eval_f1_macro": f1m}

        con.execute("CREATE SCHEMA IF NOT EXISTS ml;")
        con.register("pred_df", pred_df.to_pandas())
        con.execute("CREATE OR REPLACE TABLE ml.iris_predictions AS SELECT * FROM pred_df;")
        con.unregister("pred_df")

        context.add_output_metadata(
            {
                "rows_scored": MetadataValue.int(pred_df.height),
                "pred_table": "ml.iris_predictions",
                "model_version": version,
                **({"eval": MetadataValue.json(eval_meta)} if eval_meta else {}),
            }
        )
