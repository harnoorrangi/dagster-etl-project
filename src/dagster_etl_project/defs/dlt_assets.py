from dagster import AssetExecutionContext, AssetKey, AssetMaterialization, MetadataValue
from dagster_dlt import dlt_assets, DagsterDltResource, DagsterDltTranslator
from dagster_dlt.translator import DltResourceTranslatorData
from dagster import AssetSpec
import dlt
from dlt_sources.get_data import iris


# Make sure this matches your dbt source name/schema
iris_pipeline = dlt.pipeline(
    pipeline_name="iris_pipeline",
    dataset_name="dlt_main_dataset",
    destination="duckdb",
    progress="log",
)

TARGET_KEY = AssetKey(["dlt_main_dataset", "iris"])
GITHUB_KEY = AssetKey(["github", "iris"])

class DltIrisToDbtSourceTranslator(DagsterDltTranslator):
    def get_asset_spec(self, data: DltResourceTranslatorData) -> AssetSpec:
        spec = super().get_asset_spec(data)
        if data.resource.name in {"raw_data", "iris", "iris_raw_data"}:
            return spec.replace_attributes(
                key=TARGET_KEY,
                group_name="dlt_extract",
                tags={"source": "github"},
                deps=[GITHUB_KEY],
            )
        return spec

@dlt_assets(
    name="iris_assets",
    dlt_source=iris(),
    dlt_pipeline=iris_pipeline,
    dagster_dlt_translator=DltIrisToDbtSourceTranslator(),
)
def dagster_iris_assets(context: AssetExecutionContext, dlt_rs: DagsterDltResource):
    num_rows = 0
    for event in dlt_rs.run(context=context):
        meta = getattr(event, "metadata", None)
        if meta and "rows_loaded" in meta:
            v = meta["rows_loaded"]
            if hasattr(v, "value"):
                num_rows = v.value
                context.log.info(f"Extracted rows_loaded: {num_rows}")
        yield event

    # Attach metadata to the SAME asset key (no output_name)
    context.log_event(
        AssetMaterialization(
            asset_key=TARGET_KEY,
            metadata={"num_rows": MetadataValue.int(num_rows), "source": "dlt -> DuckDB"},
        )
    )

github_upstreams = [
    AssetSpec(
        key=key,                # matches the inferred upstream key (currently just "iris")
        group_name="Github",  # or "github" if you prefer
        tags={"source": "github", "kind": "github"},
    )
    for key in dagster_iris_assets.dependency_keys
]