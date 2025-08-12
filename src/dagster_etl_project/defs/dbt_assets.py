from pathlib import Path
from dagster import Definitions
from dagster_dbt import (
    DbtCliResource, DbtProject, dbt_assets,
    DagsterDbtTranslator,
    group_from_dbt_resource_props_fallback_to_directory,
)

DBT_DIR = Path(__file__).resolve().parents[3] / "dbt_iris"

class GroupByDirectoryTranslator(DagsterDbtTranslator):
    def get_group_name(self, dbt_resource_props):
        return group_from_dbt_resource_props_fallback_to_directory(dbt_resource_props)

dbt_project = DbtProject(project_dir=DBT_DIR)
dbt_project.prepare_if_dev() 

dbt_resource = DbtCliResource(project_dir=dbt_project)

@dbt_assets(
    manifest=dbt_project.manifest_path,
    dagster_dbt_translator=GroupByDirectoryTranslator(),
)
def dbt_iris_assets(context, dbt_resource: DbtCliResource):
    yield from dbt_resource.cli(["build"], context=context).stream()

defs = Definitions(
    assets=[dbt_iris_assets],
    resources={"dbt_resource": dbt_resource},  
)