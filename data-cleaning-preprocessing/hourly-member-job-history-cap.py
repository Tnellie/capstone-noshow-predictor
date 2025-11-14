from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    jobs_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    timeline_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    swipe_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(jobs_df, timeline_df, swipe_df):
    """
    Filter the timeline dataset to include only relevant dates for each employee and
    add job information for each day in the timeline.

    This function:
    1. Filters out dates before the actual hire date (accrualStartDate) for all employees
    2. Filters out dates after the last job date for employees whose final status is 'T' (terminated)
    3. Joins job information to each timeline day by finding the most recent job as of each date
    4. Filters to include only days where the employee was in an hourly position
    5. Preserves all original timeline features (day of week, holidays, etc.)
    """
    # Select relevant cols from jobs data
    jobs_df_selected = jobs_df.select(
        "maskedMatchId",
        "company",
        "locationId",
        "department",
        "standardDailyHours",
        "standardWeeklyHours",
        "effectiveDate",
        "managersMatchValue",
        "payType",
        "hrStatus",
        "employmentType",
        "accrualStartDate",
    )

    # Get 1st clock-in swipe for each employee to filter the timeline
    first_swipe_date_df = (
        swipe_df.filter(F.col("type") == "SWIPE_IN")
        .groupBy("maskedMatchId")
        .agg(F.min("swipeDate").alias("first_swipe_date"))
    )

    # Window to get last job record for each employee
    window_latest_job = Window.partitionBy("maskedMatchId").orderBy(
        F.desc("effectiveDate")
    )

    # Get the last job record
    latest_job_df = (
        jobs_df_selected.withColumn("rank", F.row_number().over(window_latest_job))
        .filter(F.col("rank") == 1)
        .select("maskedMatchId", "effectiveDate", "hrStatus")
        .withColumnRenamed("effectiveDate", "last_effective_date")
    )

    # Add flag for terminated employees
    latest_job_df = latest_job_df.withColumn(
        "is_terminated", F.when(F.col("hrStatus") == "T", True).otherwise(False)
    )

    # Join employment boundaries to timelines
    enhanced_timeline = timeline_df.join(
        first_swipe_date_df, "maskedMatchId", "left"
    ).join(
        latest_job_df.select("maskedMatchId", "last_effective_date", "is_terminated"),
        "maskedMatchId",
        "left",
    )

    # Keep only rows on or after first_swipe_date for all employees
    # Keep only rows on or before last_effective_date for terminated employees
    filtered_timeline = enhanced_timeline.filter(
        (F.col("date") >= F.col("first_swipe_date"))
        & (~F.col("is_terminated") | (F.col("date") <= F.col("last_effective_date")))
    )

    # Join jobs data to filtered timeline
    joined_df = filtered_timeline.alias("timeline").join(
        jobs_df_selected.alias("jobs"),
        F.col("timeline.maskedMatchId") == F.col("jobs.maskedMatchId"),
        "left",
    )

    # Helper cols for window function logic
    joined_df = joined_df.withColumn(
        "is_job_valid_for_date",
        F.when(F.col("jobs.effectiveDate") <= F.col("timeline.date"), 1).otherwise(0),
    ).withColumn(
        "date_diff",
        F.when(
            F.col("jobs.effectiveDate") <= F.col("timeline.date"),
            F.datediff(F.col("timeline.date"), F.col("jobs.effectiveDate")),
        ).otherwise(F.datediff(F.col("jobs.effectiveDate"), F.col("timeline.date"))),
    )

    # Define window to find appropriate job for each date in timeline
    window_job_selection = Window.partitionBy(
        "timeline.maskedMatchId", "timeline.date"
    ).orderBy(
        F.desc("is_job_valid_for_date"), 
        F.asc("date_diff"),
    )

    # Get all cols from timeline_df
    timeline_columns = [
        F.col("timeline." + col_name) for col_name in timeline_df.columns
    ]

    # Get all relevant cols from jobs_df
    job_columns = [
        F.col("jobs.company"),
        F.col("jobs.locationId"),
        F.col("jobs.department"),
        F.col("jobs.standardDailyHours"),
        F.col("jobs.standardWeeklyHours"),
        F.col("jobs.effectiveDate"),
        F.col("jobs.managersMatchValue"),
        F.col("jobs.payType"),
        F.col("jobs.hrStatus"),
        F.col("jobs.employmentType"),
        F.col("jobs.accrualStartDate"),
    ]

    # Assign most recent job data to each date in timeline
    result_with_job_info = (
        joined_df.withColumn("job_rank", F.row_number().over(window_job_selection))
        .filter(F.col("job_rank") == 1)
        .drop("job_rank", "first_hire_date", "last_effective_date", "is_terminated")
    )

    # Filter for hourly employees
    result_df = result_with_job_info.filter(F.col("jobs.payType") == "H").select(
        *timeline_columns, *job_columns
    )

    # Filter for full-time US members
    result_df = result_df.filter(F.col("employmentType") == "MEMBER-US")
    result_df = result_df.drop("PK", "payType", "employmentType", "effectiveDate")

    return result_df
