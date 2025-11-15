from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Creates department-level aggregate features to use as proxies for high-cardinality
    department column. Calculates rates of no-shows, sick days, vacations, and late
    arrivals over multiple time windows.
    """
    # Define time windows
    windows = [7, 14, 30, 90]

    result_df = source_df

    # Convert date to Unix timestamp- seconds since epoch
    result_df = result_df.withColumn("date_ts", F.unix_timestamp(F.col("date")))

    # Calculate dept size
    dept_counts = result_df.groupBy("date", "department", "locationId").agg(
        F.countDistinct("maskedMatchId").alias("department_size")
    )

    # Calculate rolling stats for each time window
    for window_days in windows:
        # Time-based window: look back N days (in seconds), exclude current row
        window_seconds = window_days * 86400
        window_spec = (
            Window.partitionBy("department", "locationId")
            .orderBy("date_ts")
            .rangeBetween(-window_seconds, -1)
        )

        # Calculate counts and rates with null handling
        result_df = (
            result_df.withColumn(
                f"dept_noshow_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("noShow").cast("int")).over(window_spec), F.lit(0.0)
                ),
            )
            .withColumn(
                f"dept_sickday_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("isSickTime").cast("int")).over(window_spec), F.lit(0.0)
                ),
            )
            .withColumn(
                f"dept_vacation_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("isVacationTime").cast("int")).over(window_spec),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                f"dept_late_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("arrivedLateException").cast("int")).over(window_spec),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                f"dept_early_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("leftEarlyException").cast("int")).over(window_spec),
                    F.lit(0.0),
                ),
            )
        )

        # Calculate no-show volatility with NaN and Null handling
        result_df = result_df.withColumn(
            f"dept_noshow_volatility_{window_days}d",
            F.coalesce(
                F.nanvl(
                    F.stddev(F.col("noShow").cast("int")).over(window_spec), F.lit(0.0)
                ),
                F.lit(0.0),
            ),
        )

    # Join dept size
    result_df = result_df.join(
        dept_counts, on=["date", "department", "locationId"], how="left"
    )

    # Calculate dept day-of-week no-show rate with null handling
    window_seconds = 30 * 86400
    dow_window = (
        Window.partitionBy("department", "locationId", "dayOfWeek")
        .orderBy("date_ts")
        .rangeBetween(-window_seconds, -1)
    )

    result_df = result_df.withColumn(
        "dept_noshow_rate_dow_30d",
        F.coalesce(F.avg(F.col("noShow").cast("int")).over(dow_window), F.lit(0.0)),
    )

    # Drop temp timestamp and dept cols
    result_df = result_df.drop("date_ts", "department")

    return result_df
