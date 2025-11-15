from pyspark.sql import functions as F
from transforms.api import transform, Input, Output
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    FloatType,
    IntegerType,
)


def create_pr_curve_data_df_api(pred_df, target_col="label", prob_col="probability1"):
    """
    Create precision-recall curve data for plotting in a notebook.

    Parameters:
    pred_df - DataFrame with predictions from evaluate_fm_model
    target_col - name of the target column (default: "label")
    prob_col - name of probability column for positive class (default: "probability1")

    Returns:
    PySpark DataFrames with precision, recall data and no_skill baseline
    """
    # Calculate positive class ratio for no-skill baseline
    total = pred_df.count()
    positives = pred_df.filter(F.col(target_col) == 1.0).count()
    no_skill_precision = positives / total

    # Sort predictions by descending probability
    sorted_preds = pred_df.select(target_col, prob_col).orderBy(F.col(prob_col).desc())

    # Add row number for calculating thresholds
    window = Window.orderBy(F.monotonically_increasing_id())
    sorted_preds = sorted_preds.withColumn("row_id", F.row_number().over(window))

    # Calculate cumulative metrics at each threshold
    metrics_df = sorted_preds.withColumn("true_positive", F.col(target_col)).withColumn(
        "false_positive", 1 - F.col(target_col)
    )

    # Create window for cumulative sums
    sum_window = Window.orderBy("row_id").rangeBetween(Window.unboundedPreceding, 0)

    # Calculate TP, FP at each threshold
    metrics_df = metrics_df.withColumn(
        "tp_cumulative", F.sum("true_positive").over(sum_window)
    ).withColumn("fp_cumulative", F.sum("false_positive").over(sum_window))

    # Calculate precision and recall
    metrics_df = metrics_df.withColumn(
        "precision",
        F.when(
            (F.col("tp_cumulative") + F.col("fp_cumulative")) > 0,
            F.col("tp_cumulative") / (F.col("tp_cumulative") + F.col("fp_cumulative")),
        ).otherwise(1.0),
    ).withColumn("recall", F.col("tp_cumulative") / F.lit(float(positives)))

    # Add threshold value - which is the probability value
    metrics_df = metrics_df.withColumn("threshold", F.col(prob_col))

    # Select only needed columns
    pr_curve_df = metrics_df.select("precision", "recall", "threshold").withColumn(
        "no_skill_precision", F.lit(float(no_skill_precision))
    )

    return pr_curve_df


def create_confusion_matrix_data(
    pred_df,
    target_col="label",
    prediction_col="prediction",
    threshold=0.5,
    prob_col="probability1",
):
    """
    Create confusion matrix data for plotting in a notebook

    Parameters:
    pred_df - DataFrame with predictions
    target_col - name of target column (default: "label")
    prediction_col - name of prediction column (default: "prediction")
    threshold - classification threshold for probability (default: 0.5)
    prob_col - name of probability column (default: "probability1")

    Returns:
    PySpark DataFrame with confusion matrix data
    """
    # Create binary predictions if using probability
    if prob_col in pred_df.columns:
        pred_df = pred_df.withColumn(
            "binary_prediction",
            F.when(F.col(prob_col) >= threshold, 1.0).otherwise(0.0),
        )
        pred_col = "binary_prediction"
    else:
        pred_col = prediction_col

    # Calculate confusion matrix counts
    tp = pred_df.filter((F.col(target_col) == 1.0) & (F.col(pred_col) == 1.0)).count()
    tn = pred_df.filter((F.col(target_col) == 0.0) & (F.col(pred_col) == 0.0)).count()
    fp = pred_df.filter((F.col(target_col) == 0.0) & (F.col(pred_col) == 1.0)).count()
    fn = pred_df.filter((F.col(target_col) == 1.0) & (F.col(pred_col) == 0.0)).count()

    # Create DF with confusion matrix
    spark = SparkSession.builder.getOrCreate()
    # cm_data = [(tp, fp), (fn, tn)]

    # Create confusion matrix DF
    cm_df = spark.createDataFrame(
        [(0, 0, tn), (0, 1, fp), (1, 0, fn), (1, 1, tp)],
        ["actual", "predicted", "count"],
    )

    return cm_df


def create_roc_curve_data(pred_df, target_col="label", prob_col="probability1"):
    """
    Create ROC curve data for plotting in a notebook

    Parameters:
    pred_df - DataFrame with predictions from model
    target_col - name of target column (default: "label")
    prob_col - name of probability column for positive class (default: "probability1")

    Returns:
    PySpark DataFrame with FPR, TPR and threshold data
    """
    spark = SparkSession.builder.getOrCreate()

    # Calculate positives and negatives counts
    positives = pred_df.filter(F.col(target_col) == 1.0).count()
    negatives = pred_df.filter(F.col(target_col) == 0.0).count()

    # Sort predictions by descending probability
    sorted_preds = pred_df.select(target_col, prob_col).orderBy(F.col(prob_col).desc())

    # Add row number for calculating thresholds
    window = Window.orderBy(F.monotonically_increasing_id())
    sorted_preds = sorted_preds.withColumn("row_id", F.row_number().over(window))

    # Calculate true positives and false positives at each threshold
    metrics_df = sorted_preds.withColumn("true_positive", F.col(target_col)).withColumn(
        "false_positive", F.when(F.col(target_col) == 0.0, 1.0).otherwise(0.0)
    )

    # Create window for cumulative sums
    sum_window = Window.orderBy("row_id").rangeBetween(Window.unboundedPreceding, 0)

    # Calculate TP, FP at each threshold
    metrics_df = metrics_df.withColumn(
        "tp_cumulative", F.sum("true_positive").over(sum_window)
    ).withColumn("fp_cumulative", F.sum("false_positive").over(sum_window))

    # Calculate TPR and FPR
    metrics_df = metrics_df.withColumn(
        "tpr", F.col("tp_cumulative") / F.lit(float(positives))
    ).withColumn("fpr", F.col("fp_cumulative") / F.lit(float(negatives)))

    # Add threshold value, which is the probability value
    metrics_df = metrics_df.withColumn("threshold", F.col(prob_col))

    # Create perfect points with explicit schema
    perfect_schema = StructType(
        [
            StructField("fpr", DoubleType(), True),
            StructField("tpr", DoubleType(), True),
            StructField("threshold", DoubleType(), True),
        ]
    )

    # Add perfect (0,0) and (1,1) points to ensure complete curve
    perfect_points = spark.createDataFrame(
        [(0.0, 0.0, None), (1.0, 1.0, None)], schema=perfect_schema
    )

    # Select only needed cols and union with perfect points
    roc_curve_df = metrics_df.select("fpr", "tpr", "threshold").union(perfect_points)

    # Order by FPR for proper plotting
    roc_curve_df = roc_curve_df.orderBy("fpr")

    # Calculate AUC using the BinaryClassificationMetrics class
    from pyspark.mllib.evaluation import BinaryClassificationMetrics

    # Convert to the RDD format required by BinaryClassificationMetrics
    scoreAndLabels = pred_df.select(prob_col, F.col(target_col).cast("double")).rdd.map(
        lambda row: (float(row[0]), float(row[1]))
    )

    # Create metrics object
    metrics = BinaryClassificationMetrics(scoreAndLabels)

    # Get AUC
    auc = metrics.areaUnderROC

    # Add AUC as a column
    roc_curve_df = roc_curve_df.withColumn("auc", F.lit(float(auc)))

    return roc_curve_df


def create_lift_chart_data(
    pred_df, target_col="label", prob_col="probability1", num_bins=10
):
    """
    Create lift chart data for plotting in a notebook

    Parameters:
    pred_df - DataFrame with predictions from model
    target_col - name of target column (default: "label")
    prob_col - name of probability column for positive class (default: "probability1")
    num_bins - number of bins to divide population into (default: 10)

    Returns:
    PySpark DataFrame with percentile, cumulative_gain, random_gain, and lift values
    """
    spark = SparkSession.builder.getOrCreate()

    # Calculate total positives
    total_records = pred_df.count()
    total_positives = pred_df.filter(F.col(target_col) == 1.0).count()

    # Sort by probability descending
    window = Window.orderBy(F.col(prob_col).desc())

    # Add row number to create percentiles
    sorted_df = pred_df.withColumn("row_num", F.row_number().over(window))

    # Create bins
    sorted_df = sorted_df.withColumn(
        "percentile",
        ((F.col("row_num") * 100) / F.lit(total_records)).cast(IntegerType()),
    )

    # Group by percentile and calculate positive counts
    grouped_df = sorted_df.groupBy("percentile").agg(
        F.count("*").alias("bin_count"), F.sum(F.col(target_col)).alias("bin_positives")
    )

    # Create window for cumulative calculations
    window_cum = Window.orderBy("percentile").rangeBetween(Window.unboundedPreceding, 0)

    # Calculate cumulative counts and gain
    lift_df = (
        grouped_df.withColumn("cumulative_count", F.sum("bin_count").over(window_cum))
        .withColumn("cumulative_positives", F.sum("bin_positives").over(window_cum))
        .withColumn(
            "cumulative_percent",
            (F.col("cumulative_count") * 100 / F.lit(total_records)).cast(FloatType()),
        )
        .withColumn(
            "cumulative_captured",
            (F.col("cumulative_positives") * 100 / F.lit(total_positives)).cast(
                FloatType()
            ),
        )
        .withColumn(
            "random_captured",
            F.col("cumulative_percent"),  # Random model follows the diagonal
        )
        .withColumn("lift", F.col("cumulative_captured") / F.col("cumulative_percent"))
    )

    # Ensure a record exists for each percentile 1-100
    percentiles = spark.range(0, 101, 1).toDF("percentile")

    # Join with lift data and fill missing values
    complete_lift_df = percentiles.join(lift_df, on="percentile", how="left")

    # Get percentiles from existing data
    filled_lift_df = complete_lift_df.orderBy("percentile").na.drop()

    # Add perfect model line
    # In a perfect model, all positives are captured in the first X% of predictions,
    # where X is the percentage of positives in the dataset
    positive_rate = (total_positives / total_records) * 100

    # Create perfect model DF - make sure all values are float type
    perfect_data = []
    for p in range(101):
        if float(p) <= positive_rate:
            perfect_captured = (float(p) / positive_rate) * 100.0
        else:
            perfect_captured = 100.0
        perfect_data.append((float(p), float(perfect_captured)))

    perfect_schema = StructType(
        [
            StructField("percentile", FloatType(), True),
            StructField("perfect_captured", FloatType(), True),
        ]
    )

    perfect_df = spark.createDataFrame(perfect_data, schema=perfect_schema)

    # Cast percentile to integer for joining
    perfect_df = perfect_df.withColumn(
        "percentile", F.col("percentile").cast(IntegerType())
    )

    # Join with our lift data
    result_df = filled_lift_df.join(perfect_df, on="percentile", how="left")

    # Select and order cols for final output
    final_df = result_df.select(
        "percentile",
        F.col("cumulative_captured").alias("model_cumulative_gain"),
        F.col("random_captured").alias("random_gain"),
        F.col("perfect_captured").alias("perfect_gain"),
        F.col("lift").alias("lift"),
    ).orderBy("percentile")

    return final_df


@transform(
    pr_curve_df=Output("ri.foundry.main.dataset.<your-dataset-id>"),
    conf_mat_plt_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    roc_curve_df=Output("ri.foundry.main.dataset.<your-dataset-id>"),
    lift_chart_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    source_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(source_df, pr_curve_df, conf_mat_plt_df, roc_curve_df, lift_chart_df):
    """
    Function to create dataframes for use in creating model evaluation plots.
    """
    # Read input DF
    source_df = source_df.dataframe()

    # Create PR Curve data
    pr_curve_data = create_pr_curve_data_df_api(
        pred_df=source_df, target_col="noShow_day1_target", prob_col="prediction"
    )

    # Write plot DF
    pr_curve_df.write_dataframe(pr_curve_data)

    # Create Confusion Matrix data
    conf_mat_data = create_confusion_matrix_data(
        pred_df=source_df, target_col="noShow_day1_target", prob_col="prediction"
    )

    # Write plot DF
    conf_mat_plt_df.write_dataframe(conf_mat_data)

    # Create ROC Curve data
    roc_data = create_roc_curve_data(
        pred_df=source_df, target_col="noShow_day1_target", prob_col="prediction"
    )

    # Write plot DF
    roc_curve_df.write_dataframe(roc_data)

    # Create lift chart data
    lift_chart_data = create_lift_chart_data(
        pred_df=source_df,
        target_col="noShow_day1_target",
        prob_col="prediction",
        num_bins=10,  # Creates deciles
    )
    # Write plot DF
    lift_chart_df.write_dataframe(lift_chart_data)
