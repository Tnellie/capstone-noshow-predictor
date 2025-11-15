from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Clips the training dataset to exclude records after an employee's termination or final
    inactivity period, while preserving all valid employment history including
    extended leaves of absence, vacations, and gaps with subsequent return to work.

    This transform distinguishes between:
    1. Temporary leave (employee returns to work) - keep all data
    2. Final termination/departure (no return to work) - clip after last activity

    The approach looks at the entire employee history to find their true last work
    date, then clips only the tail-end records after permanent departure.

    Logic:
    1. Find first termination date (hrStatus='T') for each employee
    2. Find the absolute last date with any work activity across entire history
    3. Use termination date if available, otherwise last work date
    4. Keep all records up to cutoff + 1 day (preserves noShow_tomorrow target)

    This ensures extended vacations, medical leaves, and other temporary absences
    don't cause premature clipping of valid training data.

    Args:
        source_df (DataFrame): Output from lag-and-clean transform with lagged features,
                               noShow_tomorrow target, and hrStatus column.

    Returns:
        DataFrame: Clipped dataset preserving all employment history through final
                   departure, excluding only post-employment timeline records.
    """
    # Define window across entire employee history
    full_history_window = Window.partitionBy("maskedMatchId").rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )

    # Find first termination date for each employee: hrStatus = T
    df_with_flags = source_df.withColumn(
        "termination_date",
        F.first(F.when(F.col("hrStatus") == "T", F.col("date")), ignorenulls=True).over(
            full_history_window
        ),
    )

    # Find absolute date with any work activity across entire history
    df_with_flags = df_with_flags.withColumn(
        "last_work_date",
        F.last(
            F.when(
                (F.col("hoursWorked_lag1") > 0) | (F.col("dayDuration_lag1") > 0),
                F.col("date"),
            ),
            ignorenulls=True,
        ).over(full_history_window),
    )

    # Determine cutoff date:
    # 1. If terminated, use termination_date
    # 2. If never terminated but inactive, use last_work_date
    # 3. If neither exists (edge case), keep all data
    df_with_flags = df_with_flags.withColumn(
        "cutoff_date",
        F.when(
            F.col("termination_date").isNotNull(), F.col("termination_date")
        ).otherwise(F.coalesce(F.col("last_work_date"), F.lit("2099-12-31"))),
    )

    # Keep records up to 1 day after the cutoff date
    # to preserve noShow_tomorrow for the last valid prediction day
    result_df = df_with_flags.filter(
        F.col("date") <= F.date_add(F.col("cutoff_date"), 1)
    )

    # Drop helper cols and maskedMatchId
    result_df = result_df.drop("termination_date", "last_work_date", "cutoff_date")

    return result_df
