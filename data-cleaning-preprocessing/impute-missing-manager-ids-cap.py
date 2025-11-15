from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Imputes missing manager IDs by using the manager ID from other employees
    in the same department, location, shift and date.
    Falls back to "EMPTY" if no manager can be imputed.
    """
    # Get manager for each department/day/location combo
    manager_by_dept = (
        source_df.filter(F.col("managersMatchValue").isNotNull())
        .groupBy("date", "locationId", "department")
        .agg(F.first("managersMatchValue").alias("imputed_manager"))
    )

    # Join managers to source DF
    result_df = source_df.join(
        manager_by_dept, on=["date", "locationId", "department"], how="left"
    )

    # Update managersMatchValue with imputed values or "EMPTY"
    result_df = result_df.withColumn(
        "managersMatchValue",
        F.when(F.col("managersMatchValue").isNotNull(), F.col("managersMatchValue"))
        .when(F.col("imputed_manager").isNotNull(), F.col("imputed_manager"))
        .otherwise(F.lit("EMPTY")),
    )

    # Drop temp col
    result_df = result_df.drop("imputed_manager")

    return result_df
