from dagster_dlt import DagsterDltResource
import dagster as dg


dlt_resource = DagsterDltResource()

@dg.definitions
def resources():
    return dg.Definitions(resources={"dlt_rs": dlt_resource})


