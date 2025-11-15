from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["EXECUTOR_MEMORY_LARGE", "DYNAMIC_ALLOCATION_ENABLED_32_64"])
@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    swipe_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    shift_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    inferred_shft_df=Input(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
)
def compute(ctx, swipe_df, shift_df, inferred_shft_df):
    """
    Process employee time clock swipe data by matching swipes to scheduled shifts
    and calculating shift metrics. Enhanced with inferred shift data for better
    processing of employees without shift schedule data.
    """
    # Get spark session from Foundry's context object
    spark = ctx.spark_session

    # Define swipe categories
    swipe_df = swipe_df.withColumn("code", F.lit(0))
    swipe_df = swipe_df.withColumn(
        "code",
        F.when((F.col("payCode") == "CLOCK") & (F.col("type") == "SWIPE_IN"), 1)
        .when((F.col("payCode") == "BREAK") & (F.col("type") == "TRANSFER_PAY"), 2)
        .when((F.col("payCode") == "CLOCK") & (F.col("type") == "TRANSFER_PAY"), 3)
        .when((F.col("payCode") == "CLOCK") & (F.col("type") == "SWIPE_OUT"), 4)
        .otherwise(F.col("code")),
    )

    # Select required cols from shift_df
    shift_schedule = shift_df.select(
        "maskedMatchId", "workDate", "startAtTimestampLocal", "endAtTimestampLocal"
    )

    # Get inferred shift data for members without data in shift_df
    inferred_shifts = inferred_shft_df.filter(F.col("shift").isNotNull()).select(
        "maskedMatchId", F.col("date").alias("workDate"), "shift"
    )

    # Create shift time boundaries lookup based on shift type
    shift_boundaries = spark.createDataFrame(
        [
            ("First", 3, 0, 15, 0),  # 3:00 AM - 3:00 PM (12-hour window)
            ("Second", 12, 0, 24, 0),  # 12:00 PM - 12:00 AM (12-hour window)
            ("Third", 20, 0, 8, 0),  # 8:00 PM - 8:00 AM (12-hour window, cross-date)
        ],
        StructType(
            [
                StructField("shift", StringType(), True),
                StructField("start_hour", IntegerType(), True),
                StructField("start_minute", IntegerType(), True),
                StructField("end_hour", IntegerType(), True),
                StructField("end_minute", IntegerType(), True),
            ]
        ),
    )

    # Create synthetic schedule
    synthetic_schedule = (
        inferred_shifts.join(shift_boundaries, on=["shift"], how="left")
        .withColumn(
            "startAtTimestampLocal",
            F.expr(
                "make_timestamp("
                "year(workDate), month(workDate), day(workDate), "
                "start_hour, start_minute, 0.0)"
            ),
        )
        .withColumn(
            "endAtTimestampLocal",
            F.when(
                F.col("end_hour") < F.col("start_hour"),  # Cross-date shift
                F.expr(
                    "make_timestamp("
                    "year(workDate), month(workDate), day(workDate) + 1, "
                    "end_hour, end_minute, 0.0)"
                ),
            ).otherwise(
                F.expr(
                    "make_timestamp("
                    "year(workDate), month(workDate), day(workDate), "
                    "end_hour, end_minute, 0.0)"
                )
            ),
        )
        .select(
            "maskedMatchId", "workDate", "startAtTimestampLocal", "endAtTimestampLocal"
        )
    )

    # Combine real shift schedule with synthetic schedule - prioritize real shifts
    employees_with_real_shifts = shift_schedule.select(
        "maskedMatchId", "workDate"
    ).distinct()

    # Create synthetic schedules for employees/dates without real shifts
    synthetic_schedule_filtered = synthetic_schedule.join(
        employees_with_real_shifts,
        on=["maskedMatchId", "workDate"],
        how="left_anti",
    )

    combined_schedule = shift_schedule.unionByName(synthetic_schedule_filtered)

    # Match clock-ins to shifts
    clockins_matched = (
        swipe_df.filter(F.col("code") == 1)
        .alias("ci")
        .join(
            combined_schedule.alias("sh"),
            (F.col("ci.maskedMatchId") == F.col("sh.maskedMatchId"))
            & (
                F.col("ci.swipeTimestampLocal")
                >= F.expr("sh.startAtTimestampLocal - INTERVAL 18 HOURS")
            )
            & (
                F.col("ci.swipeTimestampLocal")
                <= F.expr("sh.startAtTimestampLocal + INTERVAL 6 HOURS")
            ),
            "left",
        )
        .withColumn(
            "distance_from_start",
            F.coalesce(
                F.abs(
                    F.unix_timestamp("ci.swipeTimestampLocal")
                    - F.unix_timestamp("sh.startAtTimestampLocal")
                ),
                F.lit(999999),
            ),
        )
        .withColumn(
            "shift_match_rank",
            F.row_number().over(
                Window.partitionBy(
                    "ci.maskedMatchId", "ci.swipeTimestampLocal"
                ).orderBy("distance_from_start")
            ),
        )
        .filter(F.col("shift_match_rank") == 1)
        .select(
            F.col("ci.maskedMatchId").alias("maskedMatchId"),
            F.col("ci.swipeTimestampLocal").alias("clockInTime"),
            # Use shift's workDate if matched, otherwise derive from swipe time
            F.when(F.col("sh.workDate").isNotNull(), F.col("sh.workDate"))
            .otherwise(
                # For unmatched swipes: assign to previous day if before 6 AM
                F.when(
                    F.hour("ci.swipeTimestampLocal") < 6,
                    F.date_sub(F.to_date("ci.swipeTimestampLocal"), 1),
                ).otherwise(F.to_date("ci.swipeTimestampLocal"))
            )
            .alias("workDate"),
        )
    )

    # Match clock-outs to shifts - within shift start to 12 hours after shift end
    # Clock-outs can happen well after shift end for overnight shifts
    clockouts_matched = (
        swipe_df.filter(F.col("code") == 4)
        .alias("co")
        .join(
            combined_schedule.alias("sh"),
            (F.col("co.maskedMatchId") == F.col("sh.maskedMatchId"))
            & (F.col("co.swipeTimestampLocal") >= F.col("sh.startAtTimestampLocal"))
            & (
                F.col("co.swipeTimestampLocal")
                <= F.expr("sh.endAtTimestampLocal + INTERVAL 12 HOURS")
            ),
            "left",
        )
        .withColumn(
            # Distance from scheduled end
            "distance_from_end",
            F.coalesce(
                F.abs(
                    F.unix_timestamp("co.swipeTimestampLocal")
                    - F.unix_timestamp("sh.endAtTimestampLocal")
                ),
                F.lit(999999),
            ),
        )
        .withColumn(
            "shift_match_rank",
            F.row_number().over(
                Window.partitionBy(
                    "co.maskedMatchId", "co.swipeTimestampLocal"
                ).orderBy("distance_from_end")
            ),
        )
        .filter(F.col("shift_match_rank") == 1)
        .withColumn(
            # Handle orphaned clock-outs (no shift match)
            # Assign to previous day if before 6 AM
            "workDate",
            F.when(F.col("sh.workDate").isNotNull(), F.col("sh.workDate")).otherwise(
                F.when(
                    F.hour("co.swipeTimestampLocal") < 6,
                    F.date_sub(F.to_date("co.swipeTimestampLocal"), 1),
                ).otherwise(F.to_date("co.swipeTimestampLocal"))
            ),
        )
        .select(
            F.col("co.maskedMatchId").alias("maskedMatchId"),
            F.col("co.swipeTimestampLocal").alias("clockOutTime"),
            "workDate",
        )
    )

    # Match meal swipes to shifts by finding the clock-in/clock-out they fall between
    # Get all clock-in and clock-out times and ensure one pair per workDate
    clock_pairs = (
        clockins_matched.alias("ci")
        .join(
            clockouts_matched.alias("co"),
            (F.col("ci.maskedMatchId") == F.col("co.maskedMatchId"))
            & (F.col("ci.workDate") == F.col("co.workDate")),
            "outer",
        )
        .select(
            F.coalesce(F.col("ci.maskedMatchId"), F.col("co.maskedMatchId")).alias(
                "maskedMatchId"
            ),
            F.coalesce(F.col("ci.workDate"), F.col("co.workDate")).alias("workDate"),
            F.col("ci.clockInTime").alias("clockInTime"),
            F.col("co.clockOutTime").alias("clockOutTime"),
        )
        .groupBy("maskedMatchId", "workDate")
        .agg(
            F.min("clockInTime").alias("clockInTime"),
            F.max("clockOutTime").alias("clockOutTime"),
        )
    )

    # Match meal-ins: must be after clock-in and before clock-out or within reasonable time if no clock-out
    meal_ins_with_clocks = (
        swipe_df.filter(F.col("code") == 2)
        .alias("mi")
        .join(
            clock_pairs.alias("cp"),
            (F.col("mi.maskedMatchId") == F.col("cp.maskedMatchId"))
            & (
                # Meal-in must be after clock-in
                (
                    (F.col("cp.clockInTime").isNotNull())
                    & (F.col("mi.swipeTimestampLocal") >= F.col("cp.clockInTime"))
                )
                | (F.col("cp.clockInTime").isNull())
            )
            & (
                # Meal-in must be before clock-out or within 12 hours of clock-in if no clock-out
                (
                    (F.col("cp.clockOutTime").isNotNull())
                    & (F.col("mi.swipeTimestampLocal") <= F.col("cp.clockOutTime"))
                )
                | (
                    (F.col("cp.clockOutTime").isNull())
                    & (F.col("cp.clockInTime").isNotNull())
                    & (
                        F.col("mi.swipeTimestampLocal")
                        <= F.expr("cp.clockInTime + INTERVAL 16 HOURS")
                    )
                )
            ),
            "inner",
        )
        .withColumn(
            "time_after_clockin",
            F.when(
                F.col("cp.clockInTime").isNotNull(),
                F.unix_timestamp("mi.swipeTimestampLocal")
                - F.unix_timestamp("cp.clockInTime"),
            ).otherwise(999999),
        )
        .withColumn(
            "match_rank",
            F.row_number().over(
                Window.partitionBy(
                    "mi.maskedMatchId", "mi.swipeTimestampLocal"
                ).orderBy("time_after_clockin")
            ),
        )
        .filter(F.col("match_rank") == 1)
        .select(
            F.col("mi.maskedMatchId").alias("maskedMatchId"),
            F.col("cp.workDate").alias("workDate"),
            F.col("mi.swipeTimestampLocal").alias("mealInTime"),
        )
    )

    # Match meal-outs - must be after clock-in and before/near clock-out
    meal_outs_with_clocks = (
        swipe_df.filter(F.col("code") == 3)
        .alias("mo")
        .join(
            clock_pairs.alias("cp"),
            (F.col("mo.maskedMatchId") == F.col("cp.maskedMatchId"))
            & (
                # Meal-out must be after clock-in
                (
                    (F.col("cp.clockInTime").isNotNull())
                    & (F.col("mo.swipeTimestampLocal") >= F.col("cp.clockInTime"))
                )
                | (F.col("cp.clockInTime").isNull())
            )
            & (
                # Meal-out must be before clock-out or within 16 hours of clock-in if no clock-out
                (
                    (F.col("cp.clockOutTime").isNotNull())
                    & (F.col("mo.swipeTimestampLocal") <= F.col("cp.clockOutTime"))
                )
                | (
                    (F.col("cp.clockOutTime").isNull())
                    & (F.col("cp.clockInTime").isNotNull())
                    & (
                        F.col("mo.swipeTimestampLocal")
                        <= F.expr("cp.clockInTime + INTERVAL 16 HOURS")
                    )
                )
            ),
            "inner",
        )
        .withColumn(
            "time_after_clockin",
            F.when(
                F.col("cp.clockInTime").isNotNull(),
                F.unix_timestamp("mo.swipeTimestampLocal")
                - F.unix_timestamp("cp.clockInTime"),
            ).otherwise(999999),
        )
        .withColumn(
            "match_rank",
            F.row_number().over(
                Window.partitionBy(
                    "mo.maskedMatchId", "mo.swipeTimestampLocal"
                ).orderBy("time_after_clockin")
            ),
        )
        .filter(F.col("match_rank") == 1)
        .select(
            F.col("mo.maskedMatchId").alias("maskedMatchId"),
            F.col("cp.workDate").alias("workDate"),
            F.col("mo.swipeTimestampLocal").alias("mealOutTime"),
        )
    )

    # Pair meal-ins with meal-outs that come after them with validation
    # Meal swipes are matched to clock pairs; they should be on the same workDate
    meal_pairs = (
        meal_ins_with_clocks.alias("mi")
        .join(
            meal_outs_with_clocks.alias("mo"),
            (F.col("mi.maskedMatchId") == F.col("mo.maskedMatchId"))
            & (F.col("mi.workDate") == F.col("mo.workDate"))  # MUST be same workDate
            & (
                F.col("mo.mealOutTime") > F.col("mi.mealInTime")
            )  # Meal-out must be after meal-in
            & (
                # Meal duration must be reasonable: between 5 minutes and 2 hours
                (F.unix_timestamp("mo.mealOutTime") - F.unix_timestamp("mi.mealInTime"))
                >= 300  # At least 5 minutes
            )
            & (
                (F.unix_timestamp("mo.mealOutTime") - F.unix_timestamp("mi.mealInTime"))
                <= 7200  # At most 2 hours
            ),
            "inner",
        )
        .withColumn(
            "meal_pair_gap",
            F.unix_timestamp("mo.mealOutTime") - F.unix_timestamp("mi.mealInTime"),
        )
        .withColumn(
            "pair_rank",
            F.row_number().over(
                Window.partitionBy(
                    "mi.maskedMatchId", "mi.workDate", "mi.mealInTime"
                ).orderBy("meal_pair_gap")
            ),
        )
        .filter(F.col("pair_rank") == 1)
        .withColumn(
            "shift_pair_rank",
            F.row_number().over(
                Window.partitionBy("mi.maskedMatchId", "mi.workDate").orderBy(
                    "mi.mealInTime"
                )
            ),
        )
        .filter(F.col("shift_pair_rank") == 1)
        .select(
            F.col("mi.maskedMatchId").alias("maskedMatchId"),
            F.col("mi.workDate").alias("workDate"),
            F.col("mi.mealInTime").alias("mealIn"),
            F.col("mo.mealOutTime").alias("mealOut"),
        )
    )

    # Capture orphaned meal-ins
    orphaned_meal_ins = (
        meal_ins_with_clocks.join(
            meal_pairs.select(
                "maskedMatchId", "workDate", F.col("mealIn").alias("mealInTime")
            ).distinct(),
            on=["maskedMatchId", "workDate", "mealInTime"],
            how="left_anti",
        )
        .withColumn(
            "orphan_rank",
            F.row_number().over(
                Window.partitionBy("maskedMatchId", "workDate").orderBy("mealInTime")
            ),
        )
        .filter(F.col("orphan_rank") == 1)
        .select(
            "maskedMatchId",
            "workDate",
            F.col("mealInTime").alias("mealIn"),
            F.lit(None).cast("timestamp").alias("mealOut"),
        )
    )

    # Capture orphaned meal-outs
    orphaned_meal_outs = (
        meal_outs_with_clocks.join(
            meal_pairs.select(
                "maskedMatchId", "workDate", F.col("mealOut").alias("mealOutTime")
            ).distinct(),
            on=["maskedMatchId", "workDate", "mealOutTime"],
            how="left_anti",
        )
        .withColumn(
            "orphan_rank",
            F.row_number().over(
                Window.partitionBy("maskedMatchId", "workDate").orderBy("mealOutTime")
            ),
        )
        .filter(F.col("orphan_rank") == 1)
        .select(
            "maskedMatchId",
            "workDate",
            F.lit(None).cast("timestamp").alias("mealIn"),
            F.col("mealOutTime").alias("mealOut"),
        )
    )

    # Combine valid pairs with orphaned swipes
    all_meal_swipes = meal_pairs.unionByName(orphaned_meal_ins).unionByName(
        orphaned_meal_outs
    )

    # Prioritize rows with the most complete data
    meal_agg = (
        all_meal_swipes.withColumn(
            "completeness_score",
            F.when(
                F.col("mealIn").isNotNull() & F.col("mealOut").isNotNull(), 3
            )  # Both present
            .when(F.col("mealIn").isNotNull(), 2)  # Only mealIn
            .when(F.col("mealOut").isNotNull(), 1)  # Only mealOut
            .otherwise(0),
        )
        .withColumn(
            "priority_rank",
            F.row_number().over(
                Window.partitionBy("maskedMatchId", "workDate").orderBy(
                    F.desc("completeness_score")
                )
            ),
        )
        .filter(F.col("priority_rank") == 1)
        .select("maskedMatchId", "workDate", "mealIn", "mealOut")
    )

    # Combine all matched swipes by employee and work date
    # Aggregate to handle multiple swipes of the same type on the same work date
    all_dates = (
        clockins_matched.select("maskedMatchId", "workDate")
        .union(clockouts_matched.select("maskedMatchId", "workDate"))
        .union(meal_ins_with_clocks.select("maskedMatchId", "workDate"))
        .union(meal_outs_with_clocks.select("maskedMatchId", "workDate"))
        .distinct()
    )

    # Aggregate each swipe type to get earliest/latest per work date
    clockins_agg = clockins_matched.groupBy("maskedMatchId", "workDate").agg(
        F.min("clockInTime").alias("clockIn")
    )

    clockouts_agg = clockouts_matched.groupBy("maskedMatchId", "workDate").agg(
        F.max("clockOutTime").alias("clockOut")
    )

    shift_events = (
        all_dates.join(clockins_agg, on=["maskedMatchId", "workDate"], how="left")
        .join(clockouts_agg, on=["maskedMatchId", "workDate"], how="left")
        .join(meal_agg, on=["maskedMatchId", "workDate"], how="left")
        .select("maskedMatchId", "workDate", "clockIn", "mealIn", "mealOut", "clockOut")
    )

    # Validate that clock-out comes after clock-in
    # If clock-out is before clock-in, set clock-out to null
    shift_events = shift_events.withColumn(
        "clockOut",
        F.when(
            (F.col("clockIn").isNotNull())
            & (F.col("clockOut").isNotNull())
            & (F.unix_timestamp("clockOut") <= F.unix_timestamp("clockIn")),
            F.lit(None),  # Invalid (clock-out before clock-in)
        ).otherwise(F.col("clockOut")),
    )

    # Calculate raw shift duration for anomaly detection
    result_df = (
        shift_events.withColumn(
            "raw_shift_hours",
            F.when(
                F.col("clockIn").isNotNull() & F.col("clockOut").isNotNull(),
                (F.unix_timestamp("clockOut") - F.unix_timestamp("clockIn")) / 3600,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            # Create detailed anomaly type classification
            "anomalyType",
            F.when(
                # Extremely long shift (>16 hours) - likely missed clock-out from previous day
                F.col("raw_shift_hours") > 16.0,
                F.lit("extreme_long_shift"),
            )
            .when(
                # Missing clock-out (employee clocked in but never clocked out)
                F.col("clockIn").isNotNull() & F.col("clockOut").isNull(),
                F.lit("missing_clockout"),
            )
            .when(
                # Very short shift (<0.25 hours) - likely erroneous swipes
                (F.col("raw_shift_hours") > 0) & (F.col("raw_shift_hours") < 0.25),
                F.lit("very_short_shift"),
            )
            .when(
                # Meal timing anomalies - meal break longer than 2 hours
                (F.col("mealIn").isNotNull())
                & (F.col("mealOut").isNotNull())
                & (
                    (F.unix_timestamp("mealOut") - F.unix_timestamp("mealIn")) / 3600
                    > 2.0
                ),
                F.lit("excessive_meal_break"),
            )
            .when(
                # Missing clock-in but has clock-out (unusual but possible)
                (F.col("clockIn").isNull()) & (F.col("clockOut").isNotNull()),
                F.lit("missing_clockin"),
            )
            .when(
                # Has meal swipes but missing either clockIn or clockOut
                (F.col("mealIn").isNotNull() | F.col("mealOut").isNotNull())
                & (F.col("clockIn").isNull() | F.col("clockOut").isNull()),
                F.lit("meal_without_full_shift"),
            )
            .when(
                # Incomplete meal break (only mealIn or only mealOut)
                ((F.col("mealIn").isNotNull()) & (F.col("mealOut").isNull()))
                | ((F.col("mealIn").isNull()) & (F.col("mealOut").isNotNull())),
                F.lit("incomplete_meal_break"),
            )
            .when(
                # No meal break on full day (6+ hours worked, no meal swipes)
                (F.col("raw_shift_hours") >= 6.0)
                & (F.col("mealIn").isNull())
                & (F.col("mealOut").isNull()),
                F.lit("no_meal_break_full_day"),
            )
            .otherwise(F.lit("normal")),
        )
        .withColumn(
            # Simplified anomaly flag for backward compatibility
            "clockAnomaly",
            F.when(F.col("anomalyType") != "normal", F.lit(True)).otherwise(
                F.lit(False)
            ),
        )
        .withColumn(
            # Anomaly severity scoring for ML feature
            "anomalySeverity",
            F.when(F.col("anomalyType") == "extreme_long_shift", F.lit(5))
            .when(F.col("anomalyType") == "missing_clockout", F.lit(4))
            .when(F.col("anomalyType") == "missing_clockin", F.lit(4))
            .when(F.col("anomalyType") == "meal_without_full_shift", F.lit(3))
            .when(F.col("anomalyType") == "no_meal_break_full_day", F.lit(2))
            .when(F.col("anomalyType") == "excessive_meal_break", F.lit(2))
            .when(F.col("anomalyType") == "very_short_shift", F.lit(2))
            .when(F.col("anomalyType") == "incomplete_meal_break", F.lit(1))
            .otherwise(F.lit(0)),
        )
        .withColumn(
            # Clean up clockOut for anomalous long shifts
            "clockOut_cleaned",
            F.when(
                F.col("anomalyType") == "extreme_long_shift",
                F.lit(None),  # Set clockOut to null for shifts over 16 hours
            ).otherwise(F.col("clockOut")),
        )
        .withColumn(
            "dayDuration",
            F.when(
                F.col("clockIn").isNotNull() & F.col("clockOut_cleaned").isNotNull(),
                (F.unix_timestamp("clockOut_cleaned") - F.unix_timestamp("clockIn"))
                / 3600,
            )
            .when(
                F.col("clockIn").isNotNull() & F.col("clockOut_cleaned").isNull(),
                8.5,  # Default duration for missing clock-out
            )
            .otherwise(0.0),
        )
        .drop("raw_shift_hours")
        .withColumn(
            "mealDuration",
            F.when(
                F.col("mealIn").isNotNull() & F.col("mealOut").isNotNull(),
                (F.unix_timestamp("mealOut") - F.unix_timestamp("mealIn")) / 3600,
            )
            .when(
                # No meal swipes but worked a full day - assume they worked through lunch
                (F.col("mealIn").isNull() & F.col("mealOut").isNull())
                & (F.col("dayDuration") >= 6.0),
                0.0,  # No meal break taken
            )
            .when(
                # Missing one meal swipe and worked a reasonable day
                (F.col("mealIn").isNull() | F.col("mealOut").isNull())
                & (F.col("dayDuration") >= 5.0),
                0.5,  # Assume standard meal break
            )
            .otherwise(0.0),
        )
        .withColumn(
            "mealStartDiff",
            F.when(
                F.col("clockIn").isNotNull() & F.col("mealIn").isNotNull(),
                (F.unix_timestamp("mealIn") - F.unix_timestamp("clockIn")) / 3600,
            ).otherwise(0.0),
        )
        .withColumn(
            "hoursWorked",
            F.when(F.col("dayDuration") == 0, 0.0)
            .when(
                # Very short shifts - give full time
                (F.col("mealDuration") == 0) & (F.col("dayDuration") < 0.25),
                F.col("dayDuration"),
            )
            .when(
                # No meal break taken (worked through lunch) - deduct standard break
                F.col("mealDuration") == 0,
                F.greatest(F.col("dayDuration") - 0.25, F.lit(0.0)),
            )
            .otherwise(
                # Had meal break - calculate based on actual or assumed meal time
                F.greatest(
                    F.col("dayDuration") - 0.25,  # Minimum break deduction
                    F.col("dayDuration") - F.col("mealDuration"),  # Actual meal time
                    F.col("dayDuration") - F.col("mealDuration") - 0.25,  # Meal + break
                    F.col("dayDuration")
                    - F.col("mealDuration")
                    - 0.5,  # Meal + longer break
                    F.lit(0.0),
                )
            ),
        )
        .withColumn(
            # Rename clockOut_cleaned for final output
            "clockOut",
            F.col("clockOut_cleaned"),
        )
        .drop("clockOut_cleaned")
    )

    # Join with combined schedule for punctuality metrics
    result_with_schedule = (
        result_df.join(
            combined_schedule.select(
                "maskedMatchId", "workDate", "startAtTimestampLocal"
            ),
            on=["maskedMatchId", "workDate"],
            how="left",
        )
        .withColumn(
            "lateArrival",
            F.when(
                (F.col("clockIn").isNotNull())
                & (F.col("startAtTimestampLocal").isNotNull()),
                F.greatest(
                    (
                        F.unix_timestamp("clockIn")
                        - F.unix_timestamp("startAtTimestampLocal")
                    )
                    / 60,
                    F.lit(0.0),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            # Early arrival in minutes (0 if late or on-time)
            "earlyArrival",
            F.when(
                (F.col("clockIn").isNotNull())
                & (F.col("startAtTimestampLocal").isNotNull()),
                F.greatest(
                    (
                        F.unix_timestamp("startAtTimestampLocal")
                        - F.unix_timestamp("clockIn")
                    )
                    / 60,
                    F.lit(0.0),
                ),
            ).otherwise(F.lit(0.0)),
        )
    )

    # Calculate consistency metrics and time-of-day features
    consistency_window = (
        Window.partitionBy("maskedMatchId").orderBy("workDate").rowsBetween(-29, 0)
    )

    final_result = (
        result_with_schedule.withColumn(
            "clockInMinuteOfDay",
            F.when(
                F.col("clockIn").isNotNull(),
                F.hour("clockIn") * 60 + F.minute("clockIn"),
            ).otherwise(
                F.lit(-1)
            ),  # Sentinel value for missing clock-out
        )
        .withColumn(
            # Clock out minute of day
            "clockOutMinuteOfDay",
            F.when(
                F.col("clockOut").isNotNull(),
                F.hour("clockOut") * 60 + F.minute("clockOut"),
            ).otherwise(
                F.lit(-1)
            ),  # Sentinel value for missing clock-out
        )
        .withColumn(
            "clockInConsistency",
            F.when(
                F.col("clockInMinuteOfDay").isNotNull(),
                F.coalesce(
                    F.when(
                        F.isnan(
                            F.stddev("clockInMinuteOfDay").over(consistency_window)
                        ),
                        0.0,
                    ).otherwise(
                        F.stddev("clockInMinuteOfDay").over(consistency_window)
                    ),
                    F.lit(0.0),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "missedClockOut",
            (F.col("clockIn").isNotNull()) & (F.col("clockOut").isNull()),
        )
        # Anomaly rate - percentage of anomalous shifts in last 30 days
        .withColumn(
            "anomalyRate",
            F.when(
                F.count("workDate").over(consistency_window) > 0,
                F.sum(F.when(F.col("clockAnomaly"), 1).otherwise(0)).over(
                    consistency_window
                )
                / F.count("workDate").over(consistency_window),
            ).otherwise(F.lit(0.0)),
        )
        .select(
            "maskedMatchId",
            "workDate",
            "clockIn",
            "mealIn",
            "mealOut",
            "clockOut",
            "dayDuration",
            "mealDuration",
            "mealStartDiff",
            "hoursWorked",
            "lateArrival",
            "earlyArrival",
            "clockInMinuteOfDay",
            "clockOutMinuteOfDay",
            "clockInConsistency",
            "missedClockOut",
            "clockAnomaly",
            "anomalyType",
            "anomalySeverity",
            "anomalyRate",
        )
    )

    # A small number of records are duplicated for some reason
    # circle back to fix this when time permits
    final_result = final_result.dropDuplicates(["maskedMatchId", "workDate"])

    return final_result
