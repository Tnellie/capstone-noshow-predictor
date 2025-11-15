from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["EXECUTOR_MEMORY_LARGE", "DYNAMIC_ALLOCATION_ENABLED_32_64"])
@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Calculates time-off metrics for employee requests.

    Processes time-off request data and adds several derived columns:
        - firstDayDelta: Number of days between when the request was made and the start date
        - totalDays: Total number of days in the time-off request (endDate - startDate + 1)
        - dayInGroup: Position of each date within its time-off period
        - totalHours: Sum of all hours for all grouped requests

    Parameters:
    -----------
    source_df : pyspark.sql.DataFrame
        Input DataFrame containing time-off request records with the following columns:
        - maskedMatchId: Employee identifier
        - requestTimestampLocal: Timestamp when the request was made
        - requestMadeAtDate: Date portion of when the request was made
        - date: The specific day of time off
        - startDate: First day of the time-off period
        - endDate: Last day of the time-off period
        - status: Status of the request (e.g., APPROVED, REJECTED)
        - quantity: Number of hours requested for each day

    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with selcted columns
    """
    # Calculate firstDayDelta, totalDays, and dayInGroup
    source_df = (
        source_df.withColumn(
            "firstDayDelta", F.datediff(F.col("startDate"), F.col("requestMadeAtDate"))
        )
        .withColumn("totalDays", F.datediff(F.col("endDate"), F.col("startDate")) + 1)
        .withColumn("dayInGroup", F.datediff(F.col("date"), F.col("startDate")) + 1)
    )

    # Create request type flags
    source_df = (
        source_df.withColumn(
            "isSickTime", F.when(F.col("maskedCodeId") == "timeoff1", 1).otherwise(0)
        )
        .withColumn(
            "isVacationTime",
            F.when(F.col("maskedCodeId") == "timeoff2", 1).otherwise(0),
        )
        .withColumn(
            "isUnpaidTime", F.when(F.col("maskedCodeId") == "timeoff3", 1).otherwise(0)
        )
    )

    # Group by requestTimestampLocal
    window_by_request_time = Window.partitionBy(
        "maskedMatchId", "requestTimestampLocal"
    )

    # Calculate totalHours
    source_df = source_df.withColumn(
        "totalHours", F.sum("quantity").over(window_by_request_time)
    )

    source_df = source_df.select(
        "maskedMatchId",
        "requestTimestampLocal",
        "date",
        "startDate",
        "endDate",
        "status",
        "quantity",
        "firstDayDelta",
        "totalDays",
        "totalHours",
        "dayInGroup",
        "requestMadeAtDate",
        "isVacationTime",
        "isSickTime",
        "isUnpaidTime",
    )

    # Window spec to deduplicate multiple requests for the same date
    window_for_deduplication = Window.partitionBy("maskedMatchId", "date").orderBy(
        F.col("requestTimestampLocal").desc()  # Most recent request first
    )

    # Add row_number to identify most recent request
    source_df = source_df.withColumn(
        "row_num", F.row_number().over(window_for_deduplication)
    )

    # Keep most recent request for member-date combos
    deduplicated_df = source_df.filter(F.col("row_num") == 1).drop("row_num")

    # Final sort
    deduplicated_df = deduplicated_df.sort("maskedMatchId", "date", "dayInGroup")

    return deduplicated_df
