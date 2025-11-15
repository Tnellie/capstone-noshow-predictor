from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    memb_facts_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    swipe_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    shift_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(memb_facts_df, swipe_df, shift_df):
    """
    Fills missing shift data for employees using clock swipe records and other data sources.

    Addresses a common issue where employees are missing shift data in the system despite having
    clock swipe data for those days.

    Processing logic:
    1. Identifies existing real shift records that have been properly scheduled
    2. Categorizes remaining records into:
       - Planned absences (vacation, unpaid time) - shift data set to null
       - Expected work days (including unplanned absences like no-shows and sick days)
       - Other days (weekends, unscheduled days, etc.)
    3. For expected work days with missing shift data but swipe card evidence:
       - Infers shift type (First/Second/Third) based on swipe timestamp
       - Determines shift duration using similar employees in the same:
         a. Department + Location + Shift + Date (primary match)
         b. Location + Shift + Date (secondary match)
         c. Default of 8.5 hours if no matches found
    4. Preserves complete timeline for all employees across all dates

    Args:
        memb_facts_df (DataFrame): Employee facts with employee data, dates, and potentially incomplete shift info
                                  Expected to have columns: maskedMatchId, date, shift, shiftDurationHrs,
                                  isVacationTime, isUnpaidTime, noShow, isSickTime, department, locationId
        swipe_df (DataFrame): Swipe card records containing timestamps of employee building access
                             Expected to have columns: maskedMatchId, swipeDate, swipeTimestampLocal, type
        shift_df (DataFrame): Shift schedule data with official shift assignments
                             Expected to have columns: maskedMatchId, startAtTimestampLocal

    Returns:
        DataFrame: Complete employee timelines with filled shift and duration data based on business rules.
                  All dates are preserved for all employees to maintain complete time series integrity.
                  Row count matches exactly with the input memb_facts_df.
    """
    # Define shift start times
    shift_1 = 5
    shift_2 = 14
    shift_3 = 21

    # Add unique row identifier
    base_df = memb_facts_df.withColumn("row_id", F.monotonically_increasing_id())

    # Get real shifts - employees/dates that have shift data
    real_shifts = shift_df.select(
        "maskedMatchId",
        F.to_date("startAtTimestampLocal").alias("date"),
    ).distinct()

    # Aggregate swipe data: one row per employee per day
    swipe_in_data = (
        swipe_df.filter(F.col("type") == "SWIPE_IN")
        .groupBy("maskedMatchId", F.col("swipeDate").alias("date"))
        .agg(
            F.min("swipeTimestampLocal").alias(
                "swipeTimestampLocal"
            )  # First swipe of the day
        )
    )

    # Create classifier df with info needed to infer shifts
    classifier_df = (
        base_df
        # Add swipe data
        .join(swipe_in_data, on=["maskedMatchId", "date"], how="left").withColumn(
            "has_swipe_data", F.col("swipeTimestampLocal").isNotNull()
        )
        # Add real shift flag
        .join(
            real_shifts.withColumn("has_real_shift", F.lit(True)),
            on=["maskedMatchId", "date"],
            how="left",
        )
    )

    # Create flags for different categories
    classifier_df = classifier_df.withColumn(
        "category",
        # Real shifts - keep as is
        F.when(F.col("has_real_shift") == True, F.lit("real_shift"))
        # Planned absences - null shift data
        .when(
            (F.col("isVacationTime") == 1) | (F.col("isUnpaidTime") == 1),
            F.lit("planned_absence"),
        )
        # Expected work days + unplanned absences - keep/fill shift data
        .when(
            (F.col("isVacationTime") != 1)
            & (F.col("isUnpaidTime") != 1)
            & (F.col("isVacationTime").isNotNull())
            & (F.col("isUnpaidTime").isNotNull())
            & (
                (F.col("noShow") == 1)
                | (F.col("isSickTime") == 1)
                | (F.col("shift").isNotNull())
                | (F.col("has_swipe_data"))
            ),
            F.lit("expected_work"),
        )
        # Everything else: weekends, unscheduled days
        .otherwise(F.lit("other_days")),
    )

    # Classify expected work days
    classifier_df = classifier_df.withColumn(
        "work_subcat",
        F.when(
            (F.col("category") == "expected_work")
            & F.col("shift").isNotNull()
            & F.col("shiftDurationHrs").isNotNull(),
            F.lit("complete"),
        )
        .when(
            (F.col("category") == "expected_work")
            & ((F.col("shift").isNull()) | (F.col("shiftDurationHrs").isNull()))
            & F.col("has_swipe_data"),
            F.lit("needs_filling"),
        )
        .when(F.col("category") == "expected_work", F.lit("incomplete"))
        .otherwise(None),
    )

    # Count records by category
    category_counts = classifier_df.groupBy("category").count().collect()
    for row in category_counts:
        print(f"Category {row['category']}: {row['count']} records")

    work_subcat_counts = (
        classifier_df.filter(F.col("category") == "expected_work")
        .groupBy("work_subcat")
        .count()
        .collect()
    )
    for row in work_subcat_counts:
        print(f"Work subcategory {row['work_subcat']}: {row['count']} records")

    # Calculate shift based on swipe time for records that need shifts
    classifier_df = classifier_df.withColumn(
        "calculated_shift",
        F.when(
            F.col("work_subcat") == "needs_filling",
            F.when(
                (
                    F.abs(F.hour("swipeTimestampLocal") - shift_1)
                    <= F.abs(F.hour("swipeTimestampLocal") - shift_2)
                )
                & (
                    F.abs(F.hour("swipeTimestampLocal") - shift_1)
                    <= F.abs(F.hour("swipeTimestampLocal") - shift_3)
                ),
                F.lit("First"),
            )
            .when(
                (
                    F.abs(F.hour("swipeTimestampLocal") - shift_2)
                    <= F.abs(F.hour("swipeTimestampLocal") - shift_1)
                )
                & (
                    F.abs(F.hour("swipeTimestampLocal") - shift_2)
                    <= F.abs(F.hour("swipeTimestampLocal") - shift_3)
                ),
                F.lit("Second"),
            )
            .otherwise(F.lit("Third")),
        ).otherwise(None),
    ).withColumn(
        "updated_shift",
        F.when(
            F.col("work_subcat") == "needs_filling",
            F.coalesce(F.col("shift"), F.col("calculated_shift")),
        ).otherwise(F.col("shift")),
    )

    # Create duration lookup tables
    valid_durations = memb_facts_df.filter(
        F.col("shiftDurationHrs").isNotNull()
    ).select(
        F.col("department").alias("lookup_dept"),
        F.col("locationId").alias("lookup_location"),
        F.col("shift").alias("lookup_shift"),
        F.col("date").alias("lookup_date"),
        "shiftDurationHrs",
    )

    # Department + Location + Shift match
    dept_location_shift_avg = valid_durations.groupBy(
        "lookup_dept", "lookup_location", "lookup_shift", "lookup_date"
    ).agg(F.avg("shiftDurationHrs").alias("avg_duration_dept_loc_shift"))

    # Location + Shift match (fallback)
    location_shift_avg = valid_durations.groupBy(
        "lookup_location", "lookup_shift", "lookup_date"
    ).agg(F.avg("shiftDurationHrs").alias("avg_duration_loc_shift"))

    # Add duration lookups for records that need filling
    needs_filling_df = classifier_df.filter(F.col("work_subcat") == "needs_filling")

    # Join with dept lookup
    with_dept_lookup = needs_filling_df.join(
        dept_location_shift_avg,
        on=[
            needs_filling_df.department == dept_location_shift_avg.lookup_dept,
            needs_filling_df.locationId == dept_location_shift_avg.lookup_location,
            needs_filling_df.updated_shift == dept_location_shift_avg.lookup_shift,
            needs_filling_df.date == dept_location_shift_avg.lookup_date,
        ],
        how="left",
    ).select(
        needs_filling_df["*"],
        dept_location_shift_avg.avg_duration_dept_loc_shift,
    )

    # Join with location lookup
    with_location_lookup = with_dept_lookup.join(
        location_shift_avg,
        on=[
            with_dept_lookup.locationId == location_shift_avg.lookup_location,
            with_dept_lookup.updated_shift == location_shift_avg.lookup_shift,
            with_dept_lookup.date == location_shift_avg.lookup_date,
        ],
        how="left",
    ).select(with_dept_lookup["*"], location_shift_avg.avg_duration_loc_shift)

    # Create lookup map for durations
    duration_lookup = with_location_lookup.select(
        "row_id", "avg_duration_dept_loc_shift", "avg_duration_loc_shift"
    )

    # Join duration lookup back to classifier df
    classifier_df = classifier_df.join(duration_lookup, on="row_id", how="left")

    # Apply final transforms based on category
    result_df = classifier_df.withColumn(
        "final_shift",
        F.when(F.col("category") == "real_shift", F.col("shift"))
        .when(F.col("category") == "planned_absence", F.lit(None).cast("string"))
        .when(F.col("work_subcat") == "needs_filling", F.col("updated_shift"))
        .otherwise(F.col("shift")),
    ).withColumn(
        "final_shift_duration",
        F.when(F.col("category") == "real_shift", F.col("shiftDurationHrs"))
        .when(F.col("category") == "planned_absence", F.lit(None).cast("double"))
        .when(
            F.col("work_subcat") == "needs_filling",
            F.coalesce(
                F.col("shiftDurationHrs"),
                F.col("avg_duration_dept_loc_shift"),
                F.col("avg_duration_loc_shift"),
                F.lit(8.5),  # Default fallback
            ),
        )
        .otherwise(F.col("shiftDurationHrs")),
    )

    # Select original cols plus transformed ones
    final_df = result_df.select(
        *[
            col
            for col in memb_facts_df.columns
            if col not in ["shift", "shiftDurationHrs"]
        ],
        F.col("final_shift").alias("shift"),
        F.col("final_shift_duration").alias("shiftDurationHrs"),
    )

    return final_df
