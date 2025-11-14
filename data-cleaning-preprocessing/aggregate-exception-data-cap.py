from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    exceptions_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(exceptions_df):
    """
    Aggregates time exceptions data to one row per employee per date.

    Uses MAX aggregation for exception flags - if any row for that employee-date
    has an exception flagged as 1, the aggregated row will show 1.

    Handles cases where multiple exception types occur on the same date
    or where duplicate rows exist in the source data.
    """

    # Aggregate exception data by employe/date
    result_df = exceptions_df.groupBy("maskedMatchId", "workDate").agg(
        F.max("leftEarly").alias("leftEarly"),
        F.max("arrivedLate").alias("arrivedLate"),
        F.max("insufficientVacNotice").alias("insufficientVacNotice"),
        F.max("noShow").alias("noShow"),
    )

    return result_df
