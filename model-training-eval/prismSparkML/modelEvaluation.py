from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField


def evaluate_all_classes(
    pred_df: DataFrame, target_col: str, prediction_col: str = "prediction"
) -> tuple:
    """
    Calculate precision, recall, and F1 score for each class in a classification model.

    Args:
        pred_df: DataFrame with predictions
        target_col: Column name containing the true labels
        prediction_col: Column name containing the predicted labels (default: "prediction")

    Returns:
        Tuple containing:
          - DataFrame with class-specific metrics (precision, recall, F1)
          - DataFrame with confusion matrix
    """
    spark = SparkSession.builder.getOrCreate()

    # Get unique class labels
    unique_labels = [row[0] for row in pred_df.select(target_col).distinct().collect()]

    # Create base evaluator
    base_evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol=prediction_col
    )

    # Calculate metrics for each class
    class_metrics_rows = []
    for label in unique_labels:
        # Convert to float for evaluator; keep int for DF
        label_float = float(label)
        label_int = int(label)

        precision_by_label = (
            base_evaluator.setMetricName("precisionByLabel")
            .setMetricLabel(label_float)  # Evaluator needs float
            .evaluate(pred_df)
        )
        recall_by_label = (
            base_evaluator.setMetricName("recallByLabel")
            .setMetricLabel(label_float)
            .evaluate(pred_df)
        )
        f1_by_label = (
            base_evaluator.setMetricName("fMeasureByLabel")
            .setMetricLabel(label_float)
            .evaluate(pred_df)
        )

        class_metrics_rows.append(
            {
                "class_label": label_int,
                "precision": precision_by_label,
                "recall": recall_by_label,
                "f1": f1_by_label,
            }
        )

    # Create class metrics DF
    class_metrics_schema = StructType(
        [
            StructField("class_label", IntegerType(), False),
            StructField("precision", DoubleType(), True),
            StructField("recall", DoubleType(), True),
            StructField("f1", DoubleType(), True),
        ]
    )

    class_metrics_df = spark.createDataFrame(
        class_metrics_rows, schema=class_metrics_schema
    )

    # Create confusion matrix DF
    confusion_matrix_df = (
        pred_df.groupBy(target_col)
        .pivot(prediction_col)
        .count()
        .fillna(0)
        .orderBy(target_col)
    )

    return class_metrics_df, confusion_matrix_df
