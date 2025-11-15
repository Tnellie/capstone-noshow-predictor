from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["EXECUTOR_MEMORY_LARGE", "DYNAMIC_ALLOCATION_ENABLED_32_64"])
@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    shift_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(shift_df):
    """
    Calculate employee shift durations with temporal anomaly correction for hourly employees.

    This function processes employee shift data by:
    1. Categorizing shifts (First/Second/Third) based on start times
    2. Calculating shift durations and correcting temporal anomalies using interpolation
    3. Removing duplicate shift records for the same employee-date combination

    Shift Classification:
        - First Shift: Closest to 5:00 AM start time
        - Second Shift: Closest to 2:00 PM (14:00) start time
        - Third Shift: Closest to 9:00 PM (21:00) start time

    Anomaly Detection & Correction:
        - Identifies shift durations outside realistic bounds (2-16 hours)
        - Corrects anomalies using temporal interpolation from surrounding valid shifts:
            1. Average of immediate neighbors (previous + next shift)
            2. Previous shift value if next unavailable
            3. Next shift value if previous unavailable
            4. Average of wider neighbors (±2 shifts)
            5. Default to 8.5 hours as fallback

    Args:
        shift_df (DataFrame): Employee shift data containing:
            - maskedMatchId: Employee identifier
            - workDate: Date of the shift
            - startAtTimestampLocal: Local start timestamp
            - endAtTimestampLocal: Local end timestamp
            - Other timestamp and date fields (dropped during processing)

    Returns:
        DataFrame: Processed shift data with columns:
            - maskedMatchId: Employee identifier
            - workDate: Date of shift
            - startAtTimestampLocal: Local start timestamp
            - endAtTimestampLocal: Local end timestamp
            - shift: Shift category ('First', 'Second', 'Third')
            - locationId: Work location
            - department: Employee department
            - managersMatchValue: Manager identifier
            - shiftDurationHrs: Corrected shift duration in hours
    """
    # Define shift start times
    shift_1 = 5
    shift_2 = 14
    shift_3 = 21

    # Shift duration anomaly thresholds
    MIN_SHIFT_DURATION = 2.0  # Min realistic shift
    MAX_SHIFT_DURATION = 16.0  # Max realistic shift

    # Drop unused columns from shift_df
    shift_df = shift_df.drop(
        *[
            "endAtTimestampUTC",
            "startAtTimestampUTC",
            "activityId",
            "startAtTimeZone",
            "EndAtTimeZone",
            "startAtDate",
            "startAtTime",
            "endAtTime",
            "endAtDate",
            "endAtTimeZone",
        ]
    )

    # Assign shift name based on the hour in startAtTimestampLocal
    shift_df = shift_df.withColumn(
        "shift",
        F.when(
            (
                F.abs(F.hour("startAtTimestampLocal") - shift_1)
                <= F.abs(F.hour("startAtTimestampLocal") - shift_2)
            )
            & (
                F.abs(F.hour("startAtTimestampLocal") - shift_1)
                <= F.abs(F.hour("startAtTimestampLocal") - shift_3)
            ),
            F.lit("First"),
        )
        .when(
            (
                F.abs(F.hour("startAtTimestampLocal") - shift_2)
                <= F.abs(F.hour("startAtTimestampLocal") - shift_1)
            )
            & (
                F.abs(F.hour("startAtTimestampLocal") - shift_2)
                <= F.abs(F.hour("startAtTimestampLocal") - shift_3)
            ),
            F.lit("Second"),
        )
        .otherwise(F.lit("Third")),
    )

    # Calculate scheduled shift duration
    output_df = shift_df.withColumn(
        "rawShiftDurationHrs",
        (
            F.unix_timestamp(F.col("endAtTimestampLocal"))
            - F.unix_timestamp(F.col("startAtTimestampLocal"))
        )
        / 3600,
    )

    # Window for anomaly detection & correction
    temporal_window = Window.partitionBy("maskedMatchId").orderBy("workDate")

    # Get surrounding values for imputation
    output_df = (
        output_df.withColumn(
            "prev_duration", F.lag("rawShiftDurationHrs", 1).over(temporal_window)
        )
        .withColumn(
            "next_duration", F.lead("rawShiftDurationHrs", 1).over(temporal_window)
        )
        .withColumn(
            "prev2_duration", F.lag("rawShiftDurationHrs", 2).over(temporal_window)
        )
        .withColumn(
            "next2_duration", F.lead("rawShiftDurationHrs", 2).over(temporal_window)
        )
    )

    # Identify anomalies
    output_df = output_df.withColumn(
        "is_anomaly",
        (F.col("rawShiftDurationHrs") < MIN_SHIFT_DURATION)
        | (F.col("rawShiftDurationHrs") > MAX_SHIFT_DURATION),
    )

    # Correct anomalies
    output_df = output_df.withColumn(
        "shiftDurationHrs",
        F.when(
            F.col("is_anomaly"),
            # Average of immediate neighbors - most common case
            F.when(
                F.col("prev_duration").isNotNull() & F.col("next_duration").isNotNull(),
                (F.col("prev_duration") + F.col("next_duration")) / 2,
            )  # Use previous value if next is missing
            .when(
                F.col("prev_duration").isNotNull(), F.col("prev_duration")
            )  # Use next value if previous is missing
            .when(
                F.col("next_duration").isNotNull(), F.col("next_duration")
            )  # Use wider neighbors if immediate neighbors missing
            .when(
                F.col("prev2_duration").isNotNull()
                & F.col("next2_duration").isNotNull(),
                (F.col("prev2_duration") + F.col("next2_duration")) / 2,
            )
            .when(F.col("prev2_duration").isNotNull(), F.col("prev2_duration"))
            .when(
                F.col("next2_duration").isNotNull(), F.col("next2_duration")
            )  # Default to 8.5 for rare edge cases
            .otherwise(F.lit(8.5)),
        ).otherwise(
            # Valid duration lengths
            F.col("rawShiftDurationHrs")
        ),
    )

    # Drop intermediate cols
    output_df = output_df.drop(
        "rawShiftDurationHrs",
        "prev_duration",
        "next_duration",
        "prev2_duration",
        "next2_duration",
        "is_anomaly",
    )

    # Drop duplicates -> there are some errors in the data
    # where a single employee is scheduled for two shifts in first shift
    # or for a full first shift and then a record for a full second shift.
    output_df = output_df.dropDuplicates(["maskedMatchId", "workDate"])

    return output_df
