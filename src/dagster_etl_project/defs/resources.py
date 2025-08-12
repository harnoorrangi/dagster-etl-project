from dagster_dlt import DagsterDltResource

from dagster_duckdb import DuckDBResource

import dagster as dg


dlt_resource = DagsterDltResource()
database = DuckDBResource(database=dg.EnvVar("DYDESTINATION__DUCKDB__CREDENTIALS__DATABASE"))

@dg.definitions
def resources():
    return dg.Definitions(resources={"dlt_rs": dlt_resource, "database": database})

