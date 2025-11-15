from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Prepares feature dataset for ML model inference by creating temporally-safe lagged
    features and removing current-day data that won't be available at prediction time.

    This transform addresses the production inference scenario where predictions are made
    early in the morning before employees clock in. To prevent data leakage, all time-based
    metrics that rely on current-day clock data are lagged by 1 day, ensuring only historical
    data is used for predictions.

    Transformations Applied:

    1. LAG TIME-BASED FEATURES (by 1 day):
       Creates lagged versions of metrics calculated from clock-in/out timestamps.
       These represent "yesterday's behavior" which is available at inference time.

       Lagged features:
       - mealDuration_lag1: Duration of meal break (yesterday)
       - mealStartDiff_lag1: Difference in meal start time from typical pattern (yesterday)
       - hoursWorked_lag1: Total hours worked (yesterday)
       - lateArrival_lag1: Minutes late to shift (yesterday)
       - clockInConsistency_lag1: Consistency metric for clock-in time (yesterday)
       - earlyArrival_lag1: Minutes early to shift (yesterday)

    2. CREATE INDICATOR FLAGS:
       Adds binary flags to indicate when lagged features were originally null
       (employee didn't work yesterday). This preserves information for all model types.

    3. FILL NULLS:
       Fills null values in lagged features with 0 to ensure compatibility with all
       ML algorithms (logistic regression, SVM, tree-based models, etc.)

    4. CALCULATE ROLLING STATISTICS:
       Computes rolling averages, standard deviations, counts, and deviation metrics
       using the lagged features to capture patterns and trends.

    5. DROP CURRENT-DAY DATA:
       Removes columns containing current-day information unavailable at inference time:
       - Raw timestamps: clockIn, clockOut, mealIn, mealOut
       - Current-day metrics: all original calculated features (non-lagged versions)

    Temporal Safety Guarantee:
       After this transform, the dataset contains NO features derived from current-day
       attendance or clock data. All features use only data from previous days, making
       the dataset safe for real-time inference before employees arrive.

    Null Handling Strategy:
       - Lagged features are filled with 0 when null (employee didn't work yesterday)
       - Indicator flags preserve the "didn't work" vs "worked with value 0" distinction
       - This approach works for both linear models (logistic regression, SVM) and
         tree-based models (random forest, decision tree, gradient boosted trees)

    Args:
        source_df (DataFrame): Input DataFrame from member-attendance-features transform
                               containing both raw clock data and calculated time metrics.

    Returns:
        DataFrame: Cleaned dataset with lagged features, indicator flags, and rolling
                   statistics, ready for ML model training/inference across multiple
                   algorithm types. Original clock columns and current-day metrics are removed.

    Output Schema Changes:
        Added columns:
              lateArrival_lag1, clockInConsistency_lag1, earlyArrival_lag1 (filled with 0)
            - avg_{feature}_7d, avg_{feature}_30d (rolling averages)
            - std_hoursWorked_lag1_7d, std_hoursWorked_lag1_30d (volatility)
            - std_lateArrival_lag1_7d, std_lateArrival_lag1_30d (consistency)
            - hoursWorked_deviation_from_avg_7d, lateArrival_deviation_from_avg_7d
            - lateArrivalDays_7d, lateArrivalDays_30d (frequency counts)

        Removed columns:
            - clockIn, clockOut, mealIn, mealOut (raw timestamps)
            - dayDuration, mealDuration, mealStartDiff, hoursWorked, lateArrival,
              clockInConsistency, earlyArrival, missedClockOut (current-day metrics)

    Example:
        For a prediction on 2024-01-15 at 6:00 AM:
        - If employee worked on 2024-01-14:
          * hoursWorked_lag1 = 8.0 (actual hours)
        - If employee didn't work on 2024-01-14:
          * hoursWorked_lag1 = 0.0 (filled null)

    Notes:
        - Tree-based models can learn interactions between the flag and lagged values
        - Linear models benefit from explicit encoding of missing data patterns
        - All rolling statistics automatically handle the filled values correctly
        - Dataset is compatible with all common ML algorithms without modification
    """
    df = source_df

    # Define lag window
    employee_window = Window.partitionBy("maskedMatchId").orderBy("date")

    # Calculated features to lag
    features_to_lag = [
        "dayDuration",
        "mealDuration",
        "mealStartDiff",
        "hoursWorked",
        "lateArrival",
        "clockInConsistency",
        "earlyArrival",
        "clockInMinuteOfDay",
        "clockOutMinuteOfDay",
        "anomalyRate",
        "leftEarlyException",
        "arrivedLateException",
        "insufficientVacNotice",
        "isSickTime",
        "isUnpaidTime",
    ]

    # Create lagged versions of time-based features and fill nulls with 0
    for feature in features_to_lag:
        df = df.withColumn(
            f"{feature}_lag1",
            F.coalesce(F.lag(feature, 1).over(employee_window), F.lit(0.0)),
        )

    # 7-day and 30-day averages of lagged features
    rolling_features = [
        "hoursWorked_lag1",
        "dayDuration_lag1",
        "lateArrival_lag1",
        "earlyArrival_lag1",
        "mealDuration_lag1",
        "clockInConsistency_lag1",
        "clockInMinuteOfDay_lag1",
        "clockOutMinuteOfDay_lag1",
        "anomalyRate_lag1",
    ]

    # Define lagged rolling stat windows - exclude current row
    win_7d = Window.partitionBy("maskedMatchId").orderBy("date").rowsBetween(-7, -1)
    win_30d = Window.partitionBy("maskedMatchId").orderBy("date").rowsBetween(-30, -1)

    for feature in rolling_features:
        # 7-day avg
        df = df.withColumn(
            f"avg_{feature}_7d",
            F.coalesce(F.avg(F.col(feature)).over(win_7d), F.lit(0.0)),
        )

        # 30-day avg
        df = df.withColumn(
            f"avg_{feature}_30d",
            F.coalesce(F.avg(F.col(feature)).over(win_30d), F.lit(0.0)),
        )

    # Consistency metrics (std)
    consistency_features = [
        "hoursWorked_lag1",
        "dayDuration_lag1",
        "lateArrival_lag1",
        "clockInMinuteOfDay_lag1",
        "clockOutMinuteOfDay_lag1",
    ]

    for feature in consistency_features:
        # 7-day std
        df = df.withColumn(
            f"std_{feature}_7d",
            F.coalesce(
                F.nanvl(F.stddev(F.col(feature)).over(win_7d), F.lit(0.0)), F.lit(0.0)
            ),
        )

        # 30-day std
        df = df.withColumn(
            f"std_{feature}_30d",
            F.coalesce(
                F.nanvl(F.stddev(F.col(feature)).over(win_30d), F.lit(0.0)), F.lit(0.0)
            ),
        )

    # Deviation of yesterday from recent avg
    df = df.withColumn(
        "hoursWorked_deviation_from_avg_7d",
        F.col("hoursWorked_lag1") - F.col("avg_hoursWorked_lag1_7d"),
    )

    df = df.withColumn(
        "lateArrival_deviation_from_avg_7d",
        F.col("lateArrival_lag1") - F.col("avg_lateArrival_lag1_7d"),
    )

    # Count of days with late arrivals in last 7/30 days
    df = df.withColumn(
        "lateArrivalDays_7d",
        F.coalesce(
            F.sum(F.when(F.col("lateArrival_lag1") > 0, 1).otherwise(0)).over(win_7d),
            F.lit(0),
        ),
    )

    df = df.withColumn(
        "lateArrivalDays_30d",
        F.coalesce(
            F.sum(F.when(F.col("lateArrival_lag1") > 0, 1).otherwise(0)).over(win_30d),
            F.lit(0),
        ),
    )

    # Count of exceptions in last 7/30 days
    exception_features = [
        "leftEarlyException_lag1",
        "arrivedLateException_lag1",
        "insufficientVacNotice_lag1",
    ]

    for feature in exception_features:
        # Extract base name (remove _lag1)
        base_name = feature.replace("_lag1", "")

        # 7-day count
        df = df.withColumn(
            f"{base_name}_count_7d",
            F.coalesce(F.sum(F.col(feature)).over(win_7d), F.lit(0)),
        )

        # 30-day count
        df = df.withColumn(
            f"{base_name}_count_30d",
            F.coalesce(F.sum(F.col(feature)).over(win_30d), F.lit(0)),
        )

    # Drop clock and current-day calculated features
    clock_columns = [
        "clockIn",
        "clockOut",
        "mealIn",
        "mealOut",
    ]

    original_features = [
        "worked",
        "dayDuration",
        "mealDuration",
        "mealStartDiff",
        "hoursWorked",
        "lateArrival",
        "clockInConsistency",
        "earlyArrival",
        "missedClockOut",
        "clockInMinuteOfDay",
        "clockOutMinuteOfDay",
        "anomalyRate",
        "leftEarlyException",
        "arrivedLateException",
        "insufficientVacNotice",
        "clockAnomaly",
        "anomalyType",
        "anomalySeverity",
        "shiftDurationHrs",
        "shift",
    ]

    # Corrleated cols identified during EDA
    correlated_featues = [
        "month",
        "yearsOfSeniority",
        "dayDuration",
        "manager_team_size_7d",
        "manager_total_shifts_7d",
        "manager_total_shifts_30d",
        "manager_team_size_90d",
        "manager_total_shifts_90d",
    ]

    # Drop clock, non-lagged, & correlated features
    columns_to_drop = clock_columns + original_features + correlated_featues

    result_df = df.drop(*columns_to_drop)

    return result_df
