from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from transforms.api import transform_df, Input, Output


def profile_dataset(input_df, dataset_name: str = "Unknown"):
    """
    Comprehensive data profiling function for PySpark DataFrames

    Parameters:
    input_df: The PySpark DataFrame to profile
    dataset_name (str): Name identifier for the dataset

    Returns:
    DataFrame: PySpark DataFrame containing profiling results
    """

    # Get dataset dimensions
    total_rows = input_df.count()
    total_columns = len(input_df.columns)

    # Initialize results list
    profile_results = []

    # Profile each column
    for col_name in input_df.columns:
        col_data = input_df.select(col_name)

        # Col data types
        data_type = dict(input_df.dtypes)[col_name]

        # Count stats
        non_null_count = col_data.filter(F.col(col_name).isNotNull()).count()
        null_count = total_rows - non_null_count
        null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0

        # Distinct vals
        distinct_count = col_data.distinct().count()
        distinct_percentage = (
            (distinct_count / total_rows) * 100 if total_rows > 0 else 0
        )

        # Cardinality classification
        if distinct_percentage >= 95:
            cardinality_level = "Very High (95%+)"
        elif distinct_percentage >= 80:
            cardinality_level = "High (80-94%)"
        elif distinct_percentage >= 50:
            cardinality_level = "Medium (50-79%)"
        elif distinct_percentage >= 10:
            cardinality_level = "Low (10-49%)"
        else:
            cardinality_level = "Very Low (<10%)"

        # Initialize numeric stats as None
        min_val = max_val = mean_val = stddev_val = None
        outlier_count = outlier_percentage = None

        # Numeric col stats
        if data_type in ["int", "bigint", "float", "double", "decimal"]:
            try:
                stats = input_df.select(
                    F.min(col_name).alias("min_val"),
                    F.max(col_name).alias("max_val"),
                    F.avg(col_name).alias("mean_val"),
                    F.stddev(col_name).alias("stddev_val"),
                ).collect()[0]

                min_val = stats["min_val"]
                max_val = stats["max_val"]
                mean_val = round(stats["mean_val"], 4) if stats["mean_val"] else None
                stddev_val = (
                    round(stats["stddev_val"], 4) if stats["stddev_val"] else None
                )

                # Calculate IQR to detect outliers
                if min_val is not None and max_val is not None:
                    # Calculate quartiles
                    quartiles = input_df.select(
                        F.expr(f"percentile_approx({col_name}, 0.25)").alias("q1"),
                        F.expr(f"percentile_approx({col_name}, 0.75)").alias("q3"),
                    ).collect()[0]

                    q1 = quartiles["q1"]
                    q3 = quartiles["q3"]

                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)

                        # Calculate outlier stats
                        outlier_count = input_df.filter(
                            (F.col(col_name) < lower_bound)
                            | (F.col(col_name) > upper_bound)
                        ).count()
                        outlier_percentage = (
                            round((outlier_count / total_rows) * 100, 2)
                            if total_rows > 0
                            else 0
                        )
            except (ValueError, TypeError, ArithmeticError) as e:
                print(f"Could not compute numeric stats for column {col_name}: {e}")

        # String col stats
        avg_length = None
        if data_type in ["string", "varchar"]:
            try:
                avg_length = input_df.select(
                    F.avg(F.length(F.col(col_name))).alias("avg_len")
                ).collect()[0]["avg_len"]
                avg_length = F.round(avg_length, 2) if avg_length else None
            except (ValueError, TypeError, ArithmeticError) as e:
                print(f"Could not compute numeric stats for column {col_name}: {e}")

        # Most frequent value
        most_frequent_val = most_frequent_count = None
        try:
            freq_result = (
                col_data.groupBy(col_name)
                .count()
                .orderBy(F.desc("count"))
                .limit(1)
                .collect()
            )
            if freq_result:
                most_frequent_val = freq_result[0][col_name]
                most_frequent_count = freq_result[0]["count"]
        except (ValueError, TypeError, ArithmeticError) as e:
            print(f"Could not compute numeric stats for column {col_name}: {e}")

        # Set data quality flags
        has_nulls = "Yes" if null_count > 0 else "No"
        has_outliers = "Yes" if outlier_count and outlier_count > 0 else "No"
        high_cardinality = "Yes" if distinct_percentage > 80 else "No"
        potential_key = "Yes" if distinct_count == total_rows else "No"

        # Compile col results
        profile_results.append(
            {
                "dataset_name": dataset_name,
                "column_name": col_name,
                "data_type": data_type,
                "total_rows": total_rows,
                "total_cols": total_columns,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "null_percentage": round(null_percentage, 2),
                "distinct_count": distinct_count,
                "distinct_percentage": round(distinct_percentage, 2),
                "min_value": str(min_val) if min_val is not None else None,
                "max_value": str(max_val) if max_val is not None else None,
                "mean_value": mean_val,
                "std_deviation": stddev_val,
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_percentage,
                "avg_string_length": avg_length,
                "most_frequent_value": str(most_frequent_val)[:50]
                if most_frequent_val
                else None,
                "most_frequent_count": most_frequent_count,
                "has_nulls": has_nulls,
                "has_outliers": has_outliers,
                "cardinality_level": cardinality_level,
                "high_cardinality": high_cardinality,
                "potential_unique_key": potential_key,
            }
        )

    # Define results_df schema
    schema = StructType(
        [
            StructField("dataset_name", StringType(), True),
            StructField("column_name", StringType(), True),
            StructField("data_type", StringType(), True),
            StructField("total_rows", LongType(), True),
            StructField("total_cols", LongType(), True),
            StructField("non_null_count", LongType(), True),
            StructField("null_count", LongType(), True),
            StructField("null_percentage", DoubleType(), True),
            StructField("distinct_count", LongType(), True),
            StructField("distinct_percentage", DoubleType(), True),
            StructField("min_value", StringType(), True),
            StructField("max_value", StringType(), True),
            StructField("mean_value", DoubleType(), True),
            StructField("std_deviation", DoubleType(), True),
            StructField("outlier_count", LongType(), True),
            StructField("outlier_percentage", DoubleType(), True),
            StructField("avg_string_length", DoubleType(), True),
            StructField("most_frequent_value", StringType(), True),
            StructField("most_frequent_count", LongType(), True),
            StructField("has_nulls", StringType(), True),
            StructField("has_outliers", StringType(), True),
            StructField("cardinality_level", StringType(), True),
            StructField("high_cardinality", StringType(), True),
            StructField("potential_unique_key", StringType(), True),
        ]
    )

    spark = SparkSession.builder.getOrCreate()
    results_df = spark.createDataFrame(profile_results, schema)
    return results_df


@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    bank_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    bank_events_df=Input(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    employee_effective_dated_df=Input(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    employee_jobs_df=Input(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    shift_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    swipe_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    time_exceptions_df=Input(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    time_off_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(
    bank_df,
    bank_events_df,
    employee_effective_dated_df,
    employee_jobs_df,
    shift_df,
    swipe_df,
    time_exceptions_df,
    time_off_df,
):
    """
    Data profiling transform for workforce datasets.
    Profiles all input datasets and returns a combined DF with profiling results.
    """
    # Dict mapping dataset names to DFs
    datasets = {
        "bank": bank_df,
        "bank_events": bank_events_df,
        "employee_effective_dated": employee_effective_dated_df,
        "employee_jobs": employee_jobs_df,
        "shift": shift_df,
        "swipe": swipe_df,
        "time_exceptions": time_exceptions_df,
        "time_off": time_off_df,
    }

    # Profile datasets
    all_profiles = []
    for dataset_name, df in datasets.items():
        profile_df = profile_dataset(df, dataset_name)
        all_profiles.append(profile_df)

    # Union profile results
    combined_profile = all_profiles[0]
    for profile_df in all_profiles[1:]:
        combined_profile = combined_profile.union(profile_df)

    # Order by dataset name and col name
    return combined_profile.orderBy("dataset_name", "column_name")
