from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df):
    """
    Calculates temporal features and rolling statistics for employee attendance data,
    including consecutive days worked, absence patterns, holiday proximity metrics,
    sick day patterns, no-show patterns, and time-based metrics with lookback windows
    of 7, 14, 30, and 90 days.
    """
    # Identify days worked
    df = source_df.withColumn(
        "worked", F.col("clockIn").isNotNull() | F.col("clockOut").isNotNull()
    )

    # Convert date to integer days since epoch
    df = df.withColumn("date_int", F.datediff(F.col("date"), F.lit("1970-01-01")))

    # Define time windows for lookback periods
    employee_date_window = Window.partitionBy("maskedMatchId").orderBy("date_int")

    # Windows for rolling stats
    win_7d = Window.partitionBy("maskedMatchId").orderBy("date_int").rowsBetween(-7, -1)
    win_14d = (
        Window.partitionBy("maskedMatchId").orderBy("date_int").rowsBetween(-14, -1)
    )
    win_30d = (
        Window.partitionBy("maskedMatchId").orderBy("date_int").rowsBetween(-30, -1)
    )
    win_90d = (
        Window.partitionBy("maskedMatchId").orderBy("date_int").rowsBetween(-90, -1)
    )

    # Consecutive days metrics
    df = (
        df.withColumn("prev_date", F.lag("date", 1).over(employee_date_window))
        .withColumn("prev_worked", F.lag("worked", 1).over(employee_date_window))
        .withColumn("days_gap", F.datediff("date", "prev_date"))
        .withColumn(
            # New streak = first appearance, gap > 1 day, or previous day not worked
            "streak_start",
            (F.col("prev_date").isNull())
            | (F.col("days_gap") > 1)
            | (F.col("prev_worked") == False),
        )
        .withColumn(
            "streak_group",
            F.sum(F.when(F.col("streak_start"), 1).otherwise(0)).over(
                employee_date_window
            ),
        )
        .withColumn(
            # Only count days actually worked in consecutive streak
            "consecutiveDaysWorked",
            F.when(
                F.col("worked"),
                F.sum(F.when(F.col("worked"), 1).otherwise(0)).over(
                    Window.partitionBy("maskedMatchId", "streak_group").orderBy(
                        "date_int"
                    )
                ),
            ).otherwise(0),
        )
    )

    # Days since last absence & days since last work
    df = (
        df.withColumn("isAbsent", ~F.col("worked"))
        .withColumn(
            "daysSinceLastAbsence",
            F.when(F.col("isAbsent"), F.lit(0)).otherwise(
                F.coalesce(
                    F.datediff(
                        F.col("date"),
                        F.last(F.when(F.col("isAbsent"), F.col("date"))).over(
                            Window.partitionBy("maskedMatchId")
                            .orderBy("date_int")
                            .rowsBetween(Window.unboundedPreceding, -1)
                        ),
                    ),
                    F.lit(0),
                )
            ),
        )
        .withColumn(
            "daysSinceLastWork",
            F.when(F.col("worked"), F.lit(0)).otherwise(
                F.coalesce(
                    F.datediff(
                        F.col("date"),
                        F.last(F.when(F.col("worked"), F.col("date"))).over(
                            Window.partitionBy("maskedMatchId")
                            .orderBy("date_int")
                            .rowsBetween(Window.unboundedPreceding, -1)
                        ),
                    ),
                    F.lit(0),
                )
            ),
        )
    )

    # Calculate rolling work rates
    df = (
        df.withColumn(
            "workRate_7d",
            F.coalesce(
                F.sum(F.when(F.col("worked"), 1).otherwise(0)).over(win_7d)
                / F.nullif(F.count(F.lit(1)).over(win_7d), F.lit(0)),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "workRate_14d",
            F.coalesce(
                F.sum(F.when(F.col("worked"), 1).otherwise(0)).over(win_14d)
                / F.nullif(F.count(F.lit(1)).over(win_14d), F.lit(0)),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "workRate_30d",
            F.coalesce(
                F.sum(F.when(F.col("worked"), 1).otherwise(0)).over(win_30d)
                / F.nullif(F.count(F.lit(1)).over(win_30d), F.lit(0)),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "workRate_90d",
            F.coalesce(
                F.sum(F.when(F.col("worked"), 1).otherwise(0)).over(win_90d)
                / F.nullif(F.count(F.lit(1)).over(win_90d), F.lit(0)),
                F.lit(0.0),
            ),
        )
    )

    # Break execution plan after basic calculations
    df = df.localCheckpoint()

    # Rolling avg time metrics
    time_metrics = [
        "hoursWorked",
        "lateArrival",
        "earlyArrival",
        "dayDuration",
        "mealDuration",
        "clockInMinuteOfDay",
        "clockOutMinuteOfDay",
    ]

    for metric in time_metrics:
        # 7-day average - fill null with 0
        df = df.withColumn(
            f"avg_{metric}_7d",
            F.coalesce(
                F.avg(F.when(F.col("worked"), F.col(metric)).otherwise(None)).over(
                    win_7d
                ),
                F.lit(0.0),
            ),
        )

        # 30-day average - fill null with 0
        df = df.withColumn(
            f"avg_{metric}_30d",
            F.coalesce(
                F.avg(F.when(F.col("worked"), F.col(metric)).otherwise(None)).over(
                    win_30d
                ),
                F.lit(0.0),
            ),
        )

    # Calculate consistency metrics (std)
    for metric in ["clockInMinuteOfDay", "clockOutMinuteOfDay", "hoursWorked"]:
        # 7-day consistency - fill null and NaN with 0
        df = df.withColumn(
            f"consistency_{metric}_7d",
            F.when(
                F.isnan(
                    F.stddev(
                        F.when(F.col("worked"), F.col(metric)).otherwise(None)
                    ).over(win_7d)
                )
                | F.stddev(F.when(F.col("worked"), F.col(metric)).otherwise(None))
                .over(win_7d)
                .isNull(),
                F.lit(0.0),
            ).otherwise(
                F.stddev(F.when(F.col("worked"), F.col(metric)).otherwise(None)).over(
                    win_7d
                )
            ),
        )

        # 30-day consistency - fill null and NaN with 0
        df = df.withColumn(
            f"consistency_{metric}_30d",
            F.when(
                F.isnan(
                    F.stddev(
                        F.when(F.col("worked"), F.col(metric)).otherwise(None)
                    ).over(win_30d)
                )
                | F.stddev(F.when(F.col("worked"), F.col(metric)).otherwise(None))
                .over(win_30d)
                .isNull(),
                F.lit(0.0),
            ).otherwise(
                F.stddev(F.when(F.col("worked"), F.col(metric)).otherwise(None)).over(
                    win_30d
                )
            ),
        )

    # Calculate anomaly counts and rates
    df = (
        df.withColumn(
            "anomalyCount_7d",
            F.coalesce(
                F.sum(F.when(F.col("clockAnomaly"), 1).otherwise(0)).over(win_7d),
                F.lit(0),
            ),
        )
        .withColumn(
            "anomalyCount_30d",
            F.coalesce(
                F.sum(F.when(F.col("clockAnomaly"), 1).otherwise(0)).over(win_30d),
                F.lit(0),
            ),
        )
        .withColumn(
            "anomalySeverity_7d_avg",
            F.coalesce(
                F.avg(
                    F.when(F.col("clockAnomaly"), F.col("anomalySeverity")).otherwise(
                        None
                    )
                ).over(win_7d),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "anomalySeverity_30d_avg",
            F.coalesce(
                F.avg(
                    F.when(F.col("clockAnomaly"), F.col("anomalySeverity")).otherwise(
                        None
                    )
                ).over(win_30d),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "missedClockOutCount_30d",
            F.coalesce(
                F.sum(F.when(F.col("missedClockOut"), 1).otherwise(0)).over(win_30d),
                F.lit(0),
            ),
        )
    )

    # Calculate day-of-week patterns
    df = df.withColumn("dayOfWeek", F.dayofweek(F.col("date")))

    # Calculate work pattern stability metrics - weekly consistency
    df = (
        df.withColumn(
            "workedSameDayLastWeek", F.lag("worked", 7).over(employee_date_window)
        )
        .withColumn(
            "sameDayAsLastWeek",
            F.col("worked") & (F.col("workedSameDayLastWeek") == True),
        )
        .withColumn(
            "sameDayPatternCount_30d",
            F.sum(F.when(F.col("sameDayAsLastWeek"), 1).otherwise(0)).over(win_30d),
        )
        .withColumn(
            "weeklyConsistencyRate_30d",
            F.coalesce(
                F.sum(F.when(F.col("sameDayAsLastWeek"), 1).otherwise(0)).over(win_30d)
                / F.nullif(
                    F.sum(F.when(F.col("worked"), 1).otherwise(0)).over(win_30d),
                    F.lit(0),
                ),
                F.lit(0.0),
            ),
        )
    )

    # Holiday proximity metrics
    df = df.withColumn(
        "next_holiday_date",
        F.first(F.when(F.col("isHoliday") == 1, F.col("date"))).over(
            Window.partitionBy("maskedMatchId")
            .orderBy("date_int")
            .rowsBetween(1, Window.unboundedFollowing)
        ),
    ).withColumn(
        "daysUntilNextHoliday",
        F.when(
            F.col("next_holiday_date").isNotNull(),
            F.datediff(F.col("next_holiday_date"), F.col("date")),
        ).otherwise(
            999
        ),  # Large number for when no future holiday exists
    )

    # Days since previous holiday
    df = df.withColumn(
        "prev_holiday_date",
        F.last(F.when(F.col("isHoliday") == 1, F.col("date"))).over(
            Window.partitionBy("maskedMatchId")
            .orderBy("date_int")
            .rowsBetween(Window.unboundedPreceding, -1)
        ),
    ).withColumn(
        "daysSinceLastHoliday",
        F.when(
            F.col("prev_holiday_date").isNotNull(),
            F.datediff(F.col("date"), F.col("prev_holiday_date")),
        ).otherwise(
            999
        ),  # Large number for when no previous holiday exists
    )

    # Holiday proximity - min days to closest holiday before or after
    df = df.withColumn(
        "holidayProximity",
        F.least(F.col("daysUntilNextHoliday"), F.col("daysSinceLastHoliday")),
    )

    # Is day adjacent to a holiday - day before or after?
    df = df.withColumn(
        "adjacentToHoliday",
        (F.col("daysUntilNextHoliday") == 1) | (F.col("daysSinceLastHoliday") == 1),
    )

    # Measure absence patterns around holidays
    df = df.withColumn(
        "nextWorkDayIsHoliday", F.lead("isHoliday", 1).over(employee_date_window) == 1
    ).withColumn(
        "prevWorkDayWasHoliday", F.lag("isHoliday", 1).over(employee_date_window) == 1
    )

    # Calculate absence rates before/after holidays (30-day window) - fill nulls with 0
    df = df.withColumn(
        "absenceRateBeforeHoliday",
        F.coalesce(
            F.when(
                F.col("nextWorkDayIsHoliday"),
                F.sum(F.when(~F.col("worked"), 1).otherwise(0)).over(
                    Window.partitionBy("maskedMatchId", "nextWorkDayIsHoliday")
                    .orderBy("date_int")
                    .rowsBetween(-29, 0)
                )
                / F.count(F.lit(1)).over(
                    Window.partitionBy("maskedMatchId", "nextWorkDayIsHoliday")
                    .orderBy("date_int")
                    .rowsBetween(-29, 0)
                ),
            ),
            F.lit(0.0),
        ),
    )

    df = df.withColumn(
        "absenceRateAfterHoliday",
        F.coalesce(
            F.when(
                F.col("prevWorkDayWasHoliday"),
                F.sum(F.when(~F.col("worked"), 1).otherwise(0)).over(
                    Window.partitionBy("maskedMatchId", "prevWorkDayWasHoliday")
                    .orderBy("date_int")
                    .rowsBetween(-29, 0)
                )
                / F.count(F.lit(1)).over(
                    Window.partitionBy("maskedMatchId", "prevWorkDayWasHoliday")
                    .orderBy("date_int")
                    .rowsBetween(-29, 0)
                ),
            ),
            F.lit(0.0),
        ),
    )

    # Count holidays in the next 7 & 14 days
    df = df.withColumn(
        "holidaysNext7Days",
        F.sum(F.when(F.col("isHoliday") == 1, 1).otherwise(0)).over(
            Window.partitionBy("maskedMatchId").orderBy("date_int").rowsBetween(0, 6)
        ),
    ).withColumn(
        "holidaysNext14Days",
        F.sum(F.when(F.col("isHoliday") == 1, 1).otherwise(0)).over(
            Window.partitionBy("maskedMatchId").orderBy("date_int").rowsBetween(0, 13)
        ),
    )

    # Break execution plan after holiday calculations
    df = df.localCheckpoint()

    # Sick time metrics
    if "isSickTime" in source_df.columns:
        # Calculate days since last sick day
        df = df.withColumn(
            "daysSinceLastSickDay",
            F.when(F.col("isSickTime") == 1, F.lit(0)).otherwise(
                F.coalesce(
                    F.datediff(
                        F.col("date"),
                        F.last(F.when(F.col("isSickTime") == 1, F.col("date"))).over(
                            Window.partitionBy("maskedMatchId")
                            .orderBy("date_int")
                            .rowsBetween(Window.unboundedPreceding, -1)
                        ),
                    ),
                    F.lit(999),  # Default if no previous sick days
                )
            ),
        )

        # Calculate total sick days in various time windows
        df = (
            df.withColumn(
                "sickDays_7d",
                F.coalesce(
                    F.sum(F.when(F.col("isSickTime") == 1, 1).otherwise(0)).over(
                        win_7d
                    ),
                    F.lit(0),
                ),
            )
            .withColumn(
                "sickDays_30d",
                F.coalesce(
                    F.sum(F.when(F.col("isSickTime") == 1, 1).otherwise(0)).over(
                        win_30d
                    ),
                    F.lit(0),
                ),
            )
            .withColumn(
                "sickDays_90d",
                F.coalesce(
                    F.sum(F.when(F.col("isSickTime") == 1, 1).otherwise(0)).over(
                        win_90d
                    ),
                    F.lit(0),
                ),
            )
        )

        # Calculate running total of sick days for the year
        df = df.withColumn(
            "sickDaysThisYear",
            F.coalesce(
                F.sum(F.when(F.col("isSickTime") == 1, 1).otherwise(0)).over(
                    Window.partitionBy("maskedMatchId", F.year("date"))
                    .orderBy("date_int")
                    .rowsBetween(Window.unboundedPreceding, -1)
                ),
                F.lit(0),
            ),
        )

        # Calculate sick day patterns
        df = df.withColumn(
            "consecutiveSickDays",
            F.when(
                F.col("isSickTime") == 1,
                F.sum(F.when(F.col("isSickTime") == 1, 1).otherwise(0)).over(
                    Window.partitionBy(
                        "maskedMatchId",
                        F.sum(F.when(F.col("isSickTime") == 0, 1).otherwise(0)).over(
                            employee_date_window
                        ),
                    ).orderBy("date_int")
                ),
            ).otherwise(0),
        )

        # Do sick days often fall on specific week days - fill nulls with 0
        for day in range(1, 8):  # 1-7 for days of week
            df = df.withColumn(
                f"sickRate_day{day}",
                F.coalesce(
                    F.sum(
                        F.when(
                            (F.col("isSickTime") == 1) & (F.dayofweek("date") == day), 1
                        ).otherwise(0)
                    ).over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(F.when(F.dayofweek("date") == day, 1).otherwise(0)).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )

        # Does employee have a pattern of being sick before/after weekends - fill nulls with 0
        df = (
            df.withColumn(
                "sickDayBeforeWeekend",
                F.when(
                    (F.col("isSickTime") == 1) & (F.dayofweek("date") == 6), 1  # Friday
                ).otherwise(0),
            )
            .withColumn(
                "sickDayAfterWeekend",
                F.when(
                    (F.col("isSickTime") == 1) & (F.dayofweek("date") == 2), 1  # Monday
                ).otherwise(0),
            )
            .withColumn(
                "sickBeforeWeekendRate_30d",
                F.coalesce(
                    F.sum("sickDayBeforeWeekend").over(win_30d)
                    / F.nullif(
                        F.sum(F.when(F.dayofweek("date") == 6, 1).otherwise(0)).over(
                            win_30d
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "sickAfterWeekendRate_30d",
                F.coalesce(
                    F.sum("sickDayAfterWeekend").over(win_30d)
                    / F.nullif(
                        F.sum(F.when(F.dayofweek("date") == 2, 1).otherwise(0)).over(
                            win_30d
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )
        )

        # Does employee have a pattern of being sick before/after holidays
        df = df.withColumn(
            "sickDayBeforeHoliday",
            F.when(
                (F.col("isSickTime") == 1) & (F.col("daysUntilNextHoliday") == 1), 1
            ).otherwise(0),
        ).withColumn(
            "sickDayAfterHoliday",
            F.when(
                (F.col("isSickTime") == 1) & (F.col("daysSinceLastHoliday") == 1), 1
            ).otherwise(0),
        )

        # Break execution plan after sick time calculations
        df = df.localCheckpoint()

    # No-show metrics
    if "noShow" in source_df.columns:
        # Days since last no-show
        df = df.withColumn(
            "daysSinceLastNoShow",
            F.when(F.col("noShow") == 1, F.lit(0)).otherwise(
                F.coalesce(
                    F.datediff(
                        F.col("date"),
                        F.last(F.when(F.col("noShow") == 1, F.col("date"))).over(
                            Window.partitionBy("maskedMatchId")
                            .orderBy("date_int")
                            .rowsBetween(Window.unboundedPreceding, -1)
                        ),
                    ),
                    F.lit(999),  # Default if no previous no-shows
                )
            ),
        )

        # Total no-shows in various time windows
        df = (
            df.withColumn(
                "noShowCount_7d",
                F.coalesce(
                    F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(win_7d),
                    F.lit(0),
                ),
            )
            .withColumn(
                "noShowCount_30d",
                F.coalesce(
                    F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(win_30d),
                    F.lit(0),
                ),
            )
            .withColumn(
                "noShowCount_90d",
                F.coalesce(
                    F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(win_90d),
                    F.lit(0),
                ),
            )
        )

        # Running total of no-shows for the year
        df = df.withColumn(
            "noShowsThisYear",
            F.coalesce(
                F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(
                    Window.partitionBy("maskedMatchId", F.year("date"))
                    .orderBy("date_int")
                    .rowsBetween(Window.unboundedPreceding, -1)
                ),
                F.lit(0),
            ),
        )

        # Calculate no-show rates - fill nulls with 0
        df = df.withColumn(
            "noShowRate_30d",
            F.coalesce(
                F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(win_30d)
                / F.nullif(
                    F.sum(
                        F.when(F.col("worked") | (F.col("noShow") != 0), 1).otherwise(0)
                    ).over(win_30d),
                    F.lit(0),
                ),
                F.lit(0.0),
            ),
        ).withColumn(
            "noShowRate_90d",
            F.coalesce(
                F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(win_90d)
                / F.nullif(
                    F.sum(
                        F.when(F.col("worked") | (F.col("noShow") != 0), 1).otherwise(0)
                    ).over(win_90d),
                    F.lit(0),
                ),
                F.lit(0.0),
            ),
        )

        # Calculate consecutive no-shows
        df = df.withColumn(
            "noShowStreak",
            F.when(
                F.col("noShow") == 1,
                F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(
                    Window.partitionBy(
                        "maskedMatchId",
                        F.sum(F.when(F.col("noShow") == 0, 1).otherwise(0)).over(
                            employee_date_window
                        ),
                    ).orderBy("date_int")
                ),
            ).otherwise(0),
        ).withColumn(
            "maxNoShowStreak_30d",
            F.coalesce(F.max("noShowStreak").over(win_30d), F.lit(0)),
        )

        # Do no-shows tend to be on specific days of week - fill nulls with 0
        for day in range(1, 8):  # 1-7 for days of week
            df = df.withColumn(
                f"noShowRate_day{day}",
                F.coalesce(
                    F.sum(
                        F.when(
                            (F.col("noShow") == 1) & (F.dayofweek("date") == day), 1
                        ).otherwise(0)
                    ).over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(
                            F.when(
                                (F.col("worked") | (F.col("noShow") != 0))
                                & (F.dayofweek("date") == day),
                                1,
                            ).otherwise(0)
                        ).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )

        # No-show patterns around weekends - fill nulls with 0
        df = (
            df.withColumn(
                "noShowBeforeWeekend",
                F.when(
                    (F.col("noShow") == 1) & (F.dayofweek("date") == 6), 1  # Friday
                ).otherwise(0),
            )
            .withColumn(
                "noShowAfterWeekend",
                F.when(
                    (F.col("noShow") == 1) & (F.dayofweek("date") == 2), 1  # Monday
                ).otherwise(0),
            )
            .withColumn(
                "noShowBeforeWeekendRate",
                F.coalesce(
                    F.sum("noShowBeforeWeekend").over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(
                            F.when(
                                (F.col("worked") | (F.col("noShow") != 0))
                                & (F.dayofweek("date") == 6),
                                1,
                            ).otherwise(0)
                        ).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "noShowAfterWeekendRate",
                F.coalesce(
                    F.sum("noShowAfterWeekend").over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(
                            F.when(
                                (F.col("worked") | (F.col("noShow") != 0))
                                & (F.dayofweek("date") == 2),
                                1,
                            ).otherwise(0)
                        ).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )
        )

        # No-show patterns around holidays - fill nulls with 0
        df = (
            df.withColumn(
                "noShowBeforeHoliday",
                F.when(
                    (F.col("noShow") == 1) & (F.col("daysUntilNextHoliday") == 1), 1
                ).otherwise(0),
            )
            .withColumn(
                "noShowAfterHoliday",
                F.when(
                    (F.col("noShow") == 1) & (F.col("daysSinceLastHoliday") == 1), 1
                ).otherwise(0),
            )
            .withColumn(
                "noShowBeforeHolidayRate",
                F.coalesce(
                    F.sum(
                        F.when(F.col("noShowBeforeHoliday") == 1, 1).otherwise(0)
                    ).over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(
                            F.when(
                                (F.col("worked") | (F.col("noShow") != 0))
                                & (F.col("daysUntilNextHoliday") == 1),
                                1,
                            ).otherwise(0)
                        ).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "noShowAfterHolidayRate",
                F.coalesce(
                    F.sum(
                        F.when(F.col("noShowAfterHoliday") == 1, 1).otherwise(0)
                    ).over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(
                            F.when(
                                (F.col("worked") | (F.col("noShow") != 0))
                                & (F.col("daysSinceLastHoliday") == 1),
                                1,
                            ).otherwise(0)
                        ).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )
        )

        # No-show seasonality - by month - fill nulls with 0
        df = df.withColumn("monthOfYear", F.month(F.col("date")))

        for month in range(1, 13):
            df = df.withColumn(
                f"noShowRate_month{month}",
                F.coalesce(
                    F.sum(
                        F.when(
                            (F.col("noShow") == 1) & (F.col("monthOfYear") == month), 1
                        ).otherwise(0)
                    ).over(
                        Window.partitionBy("maskedMatchId").rowsBetween(
                            Window.unboundedPreceding, -1
                        )
                    )
                    / F.nullif(
                        F.sum(
                            F.when(
                                (F.col("worked") | (F.col("noShow") != 0))
                                & (F.col("monthOfYear") == month),
                                1,
                            ).otherwise(0)
                        ).over(
                            Window.partitionBy("maskedMatchId").rowsBetween(
                                Window.unboundedPreceding, -1
                            )
                        ),
                        F.lit(0),
                    ),
                    F.lit(0.0),
                ),
            )

        # Are no-shows correlated with sick time - fill nulls with False
        if "isSickTime" in source_df.columns:
            df = (
                df.withColumn(
                    "prev_sick", F.lag("isSickTime", 1).over(employee_date_window)
                )
                .withColumn(
                    "next_sick", F.lead("isSickTime", 1).over(employee_date_window)
                )
                .withColumn(
                    "sickDayFollowedByNoShow",
                    F.when(F.col("prev_sick").isNull(), F.lit(False)).otherwise(
                        (F.col("prev_sick") == 1) & (F.col("noShow") == 1)
                    ),
                )
                .withColumn(
                    "noShowFollowedBySickDay",
                    F.when(F.col("next_sick").isNull(), F.lit(False)).otherwise(
                        (F.col("noShow") == 1) & (F.col("next_sick") == 1)
                    ),
                )
                .drop("prev_sick", "next_sick")
                .withColumn(
                    "sickToNoShowRate_30d",
                    F.coalesce(
                        F.sum(
                            F.when(F.col("sickDayFollowedByNoShow"), 1).otherwise(0)
                        ).over(win_30d)
                        / F.nullif(
                            F.sum(
                                F.when(
                                    F.lag("isSickTime", 1).over(employee_date_window)
                                    == 1,
                                    1,
                                ).otherwise(0)
                            ).over(win_30d),
                            F.lit(0),
                        ),
                        F.lit(0.0),
                    ),
                )
                .withColumn(
                    "noShowToSickRate_30d",
                    F.coalesce(
                        F.sum(
                            F.when(F.col("noShowFollowedBySickDay"), 1).otherwise(0)
                        ).over(win_30d)
                        / F.nullif(
                            F.sum(F.when(F.col("noShow") == 1, 1).otherwise(0)).over(
                                win_30d
                            ),
                            F.lit(0),
                        ),
                        F.lit(0.0),
                    ),
                )
            )

    # List of result columns based on calculated metrics
    result_columns = source_df.columns + [
        "worked",
        "consecutiveDaysWorked",
        "daysSinceLastAbsence",
        "daysSinceLastWork",
        "workRate_7d",
        "workRate_14d",
        "workRate_30d",
        "workRate_90d",
        # Holiday-related columns
        "daysUntilNextHoliday",
        "daysSinceLastHoliday",
        "holidayProximity",
        "adjacentToHoliday",
        "absenceRateBeforeHoliday",
        "absenceRateAfterHoliday",
        "holidaysNext7Days",
        "holidaysNext14Days",
    ]

    # Add sick time columns
    if "isSickTime" in source_df.columns:
        sick_time_columns = [
            "daysSinceLastSickDay",
            "sickDays_7d",
            "sickDays_30d",
            "sickDays_90d",
            "sickDaysThisYear",
            "consecutiveSickDays",
            "sickRate_day1",
            "sickRate_day2",
            "sickRate_day3",
            "sickRate_day4",
            "sickRate_day5",
            "sickRate_day6",
            "sickRate_day7",
            "sickBeforeWeekendRate_30d",
            "sickAfterWeekendRate_30d",
            "sickDayBeforeHoliday",
            "sickDayAfterHoliday",
        ]
        result_columns.extend(sick_time_columns)

    # Add no-show columns
    if "noShow" in source_df.columns:
        no_show_columns = [
            "daysSinceLastNoShow",
            "noShowCount_7d",
            "noShowCount_30d",
            "noShowCount_90d",
            "noShowsThisYear",
            "noShowRate_30d",
            "noShowRate_90d",
            "noShowStreak",
            "maxNoShowStreak_30d",
            "noShowRate_day1",
            "noShowRate_day2",
            "noShowRate_day3",
            "noShowRate_day4",
            "noShowRate_day5",
            "noShowRate_day6",
            "noShowRate_day7",
            "noShowBeforeWeekend",
            "noShowAfterWeekend",
            "noShowBeforeWeekendRate",
            "noShowAfterWeekendRate",
            "noShowBeforeHoliday",
            "noShowAfterHoliday",
            "noShowBeforeHolidayRate",
            "noShowAfterHolidayRate",
        ]

        # Add monthly no-show rates
        for month in range(1, 13):
            no_show_columns.append(f"noShowRate_month{month}")

        # Add sick-noshow correlation metrics
        if "isSickTime" in source_df.columns:
            no_show_columns.extend(
                [
                    "sickDayFollowedByNoShow",
                    "noShowFollowedBySickDay",
                    "sickToNoShowRate_30d",
                    "noShowToSickRate_30d",
                ]
            )

        result_columns.extend(no_show_columns)

    # Add the calculated avg cols
    for metric in time_metrics:
        result_columns.append(f"avg_{metric}_7d")
        result_columns.append(f"avg_{metric}_30d")

    # Add consistency cols
    for metric in ["clockInMinuteOfDay", "clockOutMinuteOfDay", "hoursWorked"]:
        result_columns.append(f"consistency_{metric}_7d")
        result_columns.append(f"consistency_{metric}_30d")

    # Add anomaly cols
    result_columns.extend(
        [
            "anomalyCount_7d",
            "anomalyCount_30d",
            "anomalySeverity_7d_avg",
            "anomalySeverity_30d_avg",
            "missedClockOutCount_30d",
            "sameDayPatternCount_30d",
            "weeklyConsistencyRate_30d",
        ]
    )

    # Return the final df with all features
    # Exclude temp cols
    exclude_cols = [
        "date_int",
        "prev_date",
        "prev_worked",
        "days_gap",
        "streak_start",
        "streak_group",
        "next_holiday_date",
        "prev_holiday_date",
        "nextWorkDayIsHoliday",
        "prevWorkDayWasHoliday",
        "monthOfYear",
        "workedSameDayLastWeek",
    ]

    final_columns = [col for col in result_columns if col not in exclude_cols]
    result_df = df.select(*final_columns)

    return result_df
