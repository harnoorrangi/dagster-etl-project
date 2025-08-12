from dagster_duckdb_polars import DuckDBPolarsIOManager
import dagster as dg


duckdb_io_manager = DuckDBPolarsIOManager(database=dg.EnvVar("DESTINATION__DUCKDB__CREDENTIALS__DATABASE"), schema="ml")

@dg.definitions
def resources():
    return dg.Definitions(resources={"duckdb_io_manager": duckdb_io_manager})

