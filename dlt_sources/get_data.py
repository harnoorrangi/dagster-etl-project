import dlt
import pandas as pd
import os


@dlt.source
def iris():
    @dlt.resource(write_disposition="replace")
    def raw_data():
        df = pd.read_csv(os.getenv("IRIS_URL"))
        yield df.to_dict(orient="records")
    return raw_data