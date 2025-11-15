from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


def fill_nulls_with_fallbacks(result_df):
    """
    Fill null values in features using a hierarchical fallback strategy.

    Addresses the cold-start problem for new employees who lack historical
    data by imputing missing values using increasingly broader aggregations. The hierarchy
    ensures that predictions can still be made even for brand new team members.

    Fallback Hierarchy - in order of preference:
    1. Individual member's own historical average, if available
    2. Manager's team average (excluding current member, using same time window)
    3. Department average (excluding current member, using same time window)
    4. Global fallback of 0.0 (conservative default for rates)

    Args:
        result_df (DataFrame): PySpark DataFrame containing calculated member and manager
                               features with potential null values.

    Returns:
        DataFrame: Input DataFrame with null values filled according to the hierarchical
                   strategy. Drops temporary aggregation columns.

    Features Imputed:
        - member_timeoff_approval_rate_{7d,30d,90d}: Individual time-off approval rates
        - member_total_shifts_{7d,30d,90d}: Count of shifts worked
        - member_timeoff_approvals_{7d,30d,90d}: Count of approved time-off requests
        - member_timeoff_rejections_{7d,30d,90d}: Count of rejected time-off requests
        - manager_total_shifts_{7d,30d,90d}: Manager's team shift counts

    Additional Features Created:
        - member_is_new_7d: Binary flag (1 if member has 0 shifts in last 7 days)
        - member_is_new_30d: Binary flag (1 if member has 0 shifts in last 30 days)
    """
    member_features = [
        "member_timeoff_approval_rate_7d",
        "member_timeoff_approval_rate_30d",
        "member_timeoff_approval_rate_90d",
    ]

    # Map features to their time windows
    feature_windows = {
        "7d": 7 * 86400,
        "30d": 30 * 86400,
        "90d": 90 * 86400,
    }

    # Calculate manager-level fallback values - team avg (excludes current row)
    for feature in member_features:
        # Extract window size from feature name (e.g., "7d" from "member_timeoff_approval_rate_7d")
        window_key = feature.split("_")[-1]  # Gets "7d", "30d", or "90d"
        window_seconds = feature_windows[window_key]

        manager_avg_col = f"{feature}_manager_avg"
        manager_window = (
            Window.partitionBy("manager_id_temp", "department")
            .orderBy("date_ts")
            .rangeBetween(-window_seconds, -1)
        )

        result_df = result_df.withColumn(
            manager_avg_col, F.avg(feature).over(manager_window)
        )

        # Dept-level fallback
        dept_avg_col = f"{feature}_dept_avg"
        dept_window = (
            Window.partitionBy("department")
            .orderBy("date_ts")
            .rangeBetween(-window_seconds, -1)
        )

        result_df = result_df.withColumn(dept_avg_col, F.avg(feature).over(dept_window))

        # Fill with hierarchy: manager avg → dept avg → global avg (0.0 as last resort)
        result_df = result_df.withColumn(
            feature,
            F.coalesce(
                F.col(feature), F.col(manager_avg_col), F.col(dept_avg_col), F.lit(0.0)
            ),
        ).drop(manager_avg_col, dept_avg_col)

    # Fill null shift counts
    count_features = [
        "member_total_shifts_7d",
        "member_total_shifts_30d",
        "member_total_shifts_90d",
        "member_timeoff_approvals_7d",
        "member_timeoff_approvals_30d",
        "member_timeoff_approvals_90d",
        "member_timeoff_rejections_7d",
        "member_timeoff_rejections_30d",
        "member_timeoff_rejections_90d",
        "manager_total_shifts_7d",
        "manager_total_shifts_30d",
        "manager_total_shifts_90d",
    ]

    for feature in count_features:
        result_df = result_df.withColumn(feature, F.coalesce(F.col(feature), F.lit(0)))

    # Boolean flags to indicate new members who have null count/avg data
    result_df = result_df.withColumn(
        "member_is_new_7d", F.when(F.col("member_total_shifts_7d") == 0, 1).otherwise(0)
    ).withColumn(
        "member_is_new_30d",
        F.when(F.col("member_total_shifts_30d") == 0, 1).otherwise(0),
    )

    return result_df


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Generates member-level and manager-level rolling historical statistics for predicting
    employee attendance patterns.

    Creates features that capture both individual employee behavior and team
    dynamics under different managers. All features use backward-looking time windows that
    exclude the current row to prevent temporal leakage and ensure the transform can be
    safely used in production for real-time predictions.

    Feature Categories:
    1. MEMBER-LEVEL FEATURES (Individual Employee Patterns):
       - Total shifts: Count of shifts worked (used for cold-start detection)
       - Time-off patterns: Approval/rejection counts and approval rates

    2. MANAGER-LEVEL FEATURES (Team Dynamics):
       - Team no-show rates: Average no-show rate across manager's team
       - Team sick time rates: Average sick time rate across manager's team
       - Team size: Approximate distinct count of team members
       - Team volatility: Standard deviation of team no-show behavior
       - Manager time-off patterns: Team-level approval/rejection patterns

    3. ALL-TIME FEATURES (Cold-Start Support):
       - Total shifts worked across entire employment history
       - Total time-off approvals/rejections across entire history

    4. DERIVED FLAGS:
       - member_is_new_7d/30d: Indicates employees with no recent shift history

    Time Windows:
        All rolling features are calculated for 7-day, 30-day, and 90-day lookback periods
        using rangeBetween(-window_seconds, -1) to exclude the current row and prevent
        data leakage.

    Null Handling:
        Employs hierarchical imputation via fill_nulls_with_fallbacks():
        Individual → Manager Team Avg → Department Avg → Global Default (0.0)
        This ensures new employees and cold-start scenarios are handled gracefully.

    Args:
        source_df (DataFrame): Input DataFrame containing raw employee attendance records.

    Returns:
        DataFrame: Enhanced dataset with all calculated rolling features.
                   Drops temporary cols (date_ts, manager_id_temp) and
                   PII (managersMatchValue).
    """
    # Define time windows
    windows = [7, 30, 90]

    result_df = source_df

    # Convert date to Unix timestamp - seconds since epoch
    result_df = result_df.withColumn("date_ts", F.unix_timestamp(F.col("date")))

    # Store manager ID temporarily for use in fill_nulls function
    result_df = result_df.withColumn("manager_id_temp", F.col("managersMatchValue"))

    # Define member-level stats: individual employee patterns
    for window_days in windows:
        # Time-based window: look back N days (in seconds), exclude current row
        window_seconds = window_days * 86400
        member_window = (
            Window.partitionBy("maskedMatchId")
            .orderBy("date_ts")
            .rangeBetween(-window_seconds, -1)
        )

        # Attendance pattern
        result_df = result_df.withColumn(
            f"member_total_shifts_{window_days}d", F.count("*").over(member_window)
        )

        # Timeoff request patterns - sum the integer values (0 or 1)
        result_df = result_df.withColumn(
            f"member_timeoff_rejections_{window_days}d",
            F.sum(F.coalesce(F.col("timeOffRequestRejected"), F.lit(0))).over(
                member_window
            ),
        ).withColumn(
            f"member_timeoff_approvals_{window_days}d",
            F.sum(F.coalesce(F.col("timeOffRequestApproved"), F.lit(0))).over(
                member_window
            ),
        )

        # Timeoff approval rate
        result_df = result_df.withColumn(
            f"member_timeoff_approval_rate_{window_days}d",
            F.when(
                (
                    F.col(f"member_timeoff_approvals_{window_days}d")
                    + F.col(f"member_timeoff_rejections_{window_days}d")
                )
                > 0,
                F.col(f"member_timeoff_approvals_{window_days}d")
                / (
                    F.col(f"member_timeoff_approvals_{window_days}d")
                    + F.col(f"member_timeoff_rejections_{window_days}d")
                ),
            ).otherwise(None),
        )

    # Calculate manager-level stats (team patterns)
    for window_days in windows:
        window_seconds = window_days * 86400
        manager_window = (
            Window.partitionBy("managersMatchValue", "department")
            .orderBy("date_ts")
            .rangeBetween(-window_seconds, -1)
        )

        # Team attendance patterns
        result_df = (
            result_df.withColumn(
                f"manager_team_noshow_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("noShow").cast("int")).over(manager_window), F.lit(0.0)
                ),
            )
            .withColumn(
                f"manager_team_sicktime_rate_{window_days}d",
                F.coalesce(
                    F.avg(F.col("isSickTime").cast("int")).over(manager_window),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                f"manager_total_shifts_{window_days}d",
                F.coalesce(F.count("*").over(manager_window), F.lit(0)),
            )
        )

        # Team size
        result_df = result_df.withColumn(
            f"manager_team_size_{window_days}d",
            F.coalesce(
                F.approx_count_distinct("maskedMatchId").over(manager_window), F.lit(0)
            ),
        )

        # Team timeoff request patterns
        result_df = result_df.withColumn(
            f"manager_timeoff_rejections_{window_days}d",
            F.coalesce(
                F.sum(F.coalesce(F.col("timeOffRequestRejected"), F.lit(0))).over(
                    manager_window
                ),
                F.lit(0),
            ),
        ).withColumn(
            f"manager_timeoff_approvals_{window_days}d",
            F.coalesce(
                F.sum(F.coalesce(F.col("timeOffRequestApproved"), F.lit(0))).over(
                    manager_window
                ),
                F.lit(0),
            ),
        )

        # Manager timeoff approval rate - null when no requests
        result_df = result_df.withColumn(
            f"manager_timeoff_approval_rate_{window_days}d",
            F.when(
                (
                    F.col(f"manager_timeoff_approvals_{window_days}d")
                    + F.col(f"manager_timeoff_rejections_{window_days}d")
                )
                > 0,
                F.col(f"manager_timeoff_approvals_{window_days}d")
                / (
                    F.col(f"manager_timeoff_approvals_{window_days}d")
                    + F.col(f"manager_timeoff_rejections_{window_days}d")
                ),
            ).otherwise(F.lit(0.0)),
        )

        # Team volatility with NaN and null handling
        result_df = result_df.withColumn(
            f"manager_team_noshow_std_{window_days}d",
            F.coalesce(
                F.nanvl(
                    F.stddev(F.col("noShow").cast("int")).over(manager_window),
                    F.lit(0.0),
                ),
                F.lit(0.0),
            ),
        )
    # Calculate all-time member stats for cold-start scenarios
    member_alltime_window = (
        Window.partitionBy("maskedMatchId")
        .orderBy("date_ts")
        .rangeBetween(Window.unboundedPreceding, -1)
    )

    result_df = (
        result_df.withColumn(
            "member_total_shifts_alltime",
            F.coalesce(F.count("*").over(member_alltime_window), F.lit(0)),
        )
        .withColumn(
            "member_timeoff_approvals_alltime",
            F.coalesce(
                F.sum(F.coalesce(F.col("timeOffRequestApproved"), F.lit(0))).over(
                    member_alltime_window
                ),
                F.lit(0),
            ),
        )
        .withColumn(
            "member_timeoff_rejections_alltime",
            F.coalesce(
                F.sum(F.coalesce(F.col("timeOffRequestRejected"), F.lit(0))).over(
                    member_alltime_window
                ),
                F.lit(0),
            ),
        )
    )

    # Fill nulls with avgs
    result_df = fill_nulls_with_fallbacks(result_df)

    # Drop temporary columns and PII
    result_df = result_df.drop("managersMatchValue", "manager_id_temp", "date_ts")

    return result_df
