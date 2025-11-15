from pyspark.sql import functions as F
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StructType,
    StructField,
    StringType,
)
from prismSparkML.modelEvaluation import evaluate_all_classes
from prismSparkML.modelingSupport import random_parameter_search


def CV_LogisticRegression(
    cv_df,
    target_col,
    isClassifier,
    pipeline_stages,
    weightCol=None,
    fold_col="fold_id",
    id_col="record",
    date_col="evaluation_date",
    n_param_combos=50,
    param_overrides=None,
):
    """
    This script does a cross-validation hyperparameter search through various Logistic Regression Settings

    Note: For classification, this supports both BINARY and MULTICLASS classification.
    For regression, LogisticRegression is not applicable (will raise an error).

    cv_df - dataframe with:
    id_col - The record ID (default: "record")
    date_col - the date associated with the data and record (default: "evaluation_date")
    target_col - the column with the ML target
    fold_col - a column with 0-indexed folds for training data (default: "fold_id")
    "final_features" - input feature vector processed by a pipeline

    isClassifier - boolean flag: True for classification, False for regression
    weightCol - (optional) name of the column containing instance weights for training
    fold_col - name of the fold column (default: "fold_id")
    id_col - name of the ID column (default: "record")
    date_col - name of the date column (default: "evaluation_date")
    n_param_combos - (int) number of random hyperparameter combinations to test during cross-validation
    param_overrides - (optional) dict to override specific parameter values
                    Example: {"solver": ["l-bfgs"], "regParam": [0.01, 0.1]}
    """
    # Validate this is a classification task
    if not isClassifier:
        raise ValueError(
            "LogisticRegression is only applicable for classification tasks. "
            "Set isClassifier=True or use a different algorithm for regression."
        )

    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(cv_df)
    validation_df = full_pipeline.transform(cv_df)

    # Get number of folds
    K = validation_df.agg(max(col(fold_col)).alias("max_value")).collect()[0][
        "max_value"
    ]

    # Is this binary or multiclass classification?
    unique_labels = validation_df.select(target_col).distinct().collect()
    label_values = sorted([row[0] for row in unique_labels])
    is_binary = len(label_values) == 2

    # Define parameter space for Logistic Regression
    param_space = {
        "regParam": [
            0.0,
            0.001,
            0.01,
            0.1,
            0.5,
            1.0,
            5.0,
            10.0,
            50.0,
            100.0,
        ],  # Regularization parameter
        "elasticNetParam": [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
        ],  # ElasticNet mixing parameter
        "maxIter": [10, 50, 100, 200, 500, 1000, 2000],  # Max number of iterations
        "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],  # Convergence tolerance
        "fitIntercept": [True, False],  # Whether to fit intercept
        "standardization": [True, False],  # Whether to standardize features
        "threshold": [
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]
        if is_binary
        else [None],  # Decision threshold (binary only)
        "aggregationDepth": [2, 5, 10],  # Depth for tree aggregation
    }

    # Apply param space overrides if provided
    if param_overrides:
        for param, values in param_overrides.items():
            if param in param_space:
                # Ensure values is a list for consistency
                if not isinstance(values, list):
                    values = [values]
                param_space[param] = values
                print(f"Override applied: {param} = {values}")
            else:
                print(
                    f"Warning: Parameter '{param}' not found in param_space, ignoring override"
                )

    # Calculate total combos and select a random subset
    total_combinations = 1
    for values in param_space.values():
        total_combinations *= len(values)

    # Use n_param_combos random param value combos or 10% of total, whichever is smaller
    num_random_combinations = min(n_param_combos, math.ceil(total_combinations * 0.1))

    # Use the random parameter search function
    hp_list = random_parameter_search(
        param_space=param_space, num_samples=num_random_combinations, seed=42
    )

    val_performance = []

    for hp_dict in hp_list:
        # Create LogisticRegression model with hyperparams
        lr_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "regParam": hp_dict["regParam"],
            "elasticNetParam": hp_dict["elasticNetParam"],
            "maxIter": hp_dict["maxIter"],
            "tol": hp_dict["tol"],
            "fitIntercept": hp_dict["fitIntercept"],
            "standardization": hp_dict["standardization"],
            "aggregationDepth": hp_dict["aggregationDepth"],
        }

        # Add weightCol if provided
        if weightCol is not None:
            lr_params["weightCol"] = weightCol

        # Add threshold for binary clf
        if is_binary and hp_dict["threshold"] is not None:
            lr_params["threshold"] = hp_dict["threshold"]

        lr = LogisticRegression(**lr_params)

        # Cross-validation loop
        for k in range(1, K + 1):  # Start from 1, not 0
            validation_train = validation_df.filter(
                validation_df[fold_col].isin([i for i in range(1, K + 1) if i != k])
            )
            validation_test = validation_df.filter(validation_df[fold_col] == k)

            # Skip empty folds
            if validation_test.rdd.isEmpty():
                continue

            lr_model = lr.fit(validation_train)
            pred_df = lr_model.transform(validation_test)

            # Add probability cols for clf
            pred_df = pred_df.withColumn(
                "probability_array", vector_to_array(col("probability"))
            )

            if is_binary:
                pred_df = pred_df.withColumn(
                    "probability1", col("probability_array")[1]
                )
                pred_df = pred_df.select(
                    id_col,
                    date_col,
                    target_col,
                    "prediction",
                    "probability1",
                    "probability_array",
                    "probability",
                    "rawPrediction",
                )

                # Check for both classes before calculating binary metrics
                distinct_labels = pred_df.select(target_col).distinct().count()
                distinct_preds = pred_df.select("prediction").distinct().count()

                if distinct_labels >= 2 and distinct_preds >= 2:
                    # Binary clf metrics using rawPrediction vector
                    binary_eval = BinaryClassificationEvaluator(
                        labelCol=target_col,
                        rawPredictionCol="rawPrediction",
                        metricName="areaUnderROC",
                    )
                    auROC = binary_eval.evaluate(pred_df)
                    binary_eval.setMetricName("areaUnderPR")
                    auPR = binary_eval.evaluate(pred_df)
                else:
                    # Skip binary metrics if only one class present
                    auROC = float("nan")
                    auPR = float("nan")
            else:
                # For multiclass, select relevant cols (no probability1 for multiclass)
                pred_df = pred_df.select(
                    id_col,
                    date_col,
                    target_col,
                    "prediction",
                    "probability_array",
                    "probability",
                    "rawPrediction",
                )
                # Set binary metrics to NaN for multiclass
                auROC = float("nan")
                auPR = float("nan")

            # Multiclass metrics (work for both binary and multiclass)
            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol=target_col,
                predictionCol="prediction",
                metricName="accuracy",
            )
            accuracy = evaluator_accuracy.evaluate(pred_df)
            evaluator_precision = MulticlassClassificationEvaluator(
                labelCol=target_col,
                predictionCol="prediction",
                metricName="weightedPrecision",
            )
            precision = evaluator_precision.evaluate(pred_df)
            evaluator_recall = MulticlassClassificationEvaluator(
                labelCol=target_col,
                predictionCol="prediction",
                metricName="weightedRecall",
            )
            recall = evaluator_recall.evaluate(pred_df)
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol=target_col, predictionCol="prediction", metricName="f1"
            )
            f1 = evaluator_f1.evaluate(pred_df)

            val_performance.append(
                {
                    "regParam": hp_dict["regParam"],
                    "elasticNetParam": hp_dict["elasticNetParam"],
                    "maxIter": hp_dict["maxIter"],
                    "tol": hp_dict["tol"],
                    "fitIntercept": hp_dict["fitIntercept"],
                    "standardization": hp_dict["standardization"],
                    "threshold": hp_dict["threshold"],
                    "aggregationDepth": hp_dict["aggregationDepth"],
                    "fold": k,
                    "auROC": auROC if auROC is not None else float("nan"),
                    "auPR": auPR if auPR is not None else float("nan"),
                    "accuracy": accuracy if accuracy is not None else float("nan"),
                    "precision": precision if precision is not None else float("nan"),
                    "recall": recall if recall is not None else float("nan"),
                    "f1": f1 if f1 is not None else float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                }
            )

    # Define the output schema
    schema = StructType(
        [
            StructField("regParam", DoubleType(), True),
            StructField("elasticNetParam", DoubleType(), True),
            StructField("maxIter", IntegerType(), True),
            StructField("tol", DoubleType(), True),
            StructField("fitIntercept", StringType(), True),
            StructField("standardization", StringType(), True),
            StructField("threshold", DoubleType(), True),
            StructField("aggregationDepth", IntegerType(), True),
            StructField("fold", IntegerType(), True),
            StructField("auROC", DoubleType(), True),
            StructField("auPR", DoubleType(), True),
            StructField("accuracy", DoubleType(), True),
            StructField("precision", DoubleType(), True),
            StructField("recall", DoubleType(), True),
            StructField("f1", DoubleType(), True),
            StructField("rmse", DoubleType(), True),
            StructField("mae", DoubleType(), True),
            StructField("r2", DoubleType(), True),
        ]
    )

    # Convert Booleans to strings for DF compatibility
    for performance in val_performance:
        performance["fitIntercept"] = str(performance["fitIntercept"])
        performance["standardization"] = str(performance["standardization"])

    spark = SparkSession.builder.getOrCreate()
    df_performance = spark.createDataFrame(val_performance, schema=schema)
    return df_performance


def train_lr_model(
    training_df, params, target_col, isClassifier, pipeline_stages, weightCol=None
):
    """
    Train a Logistic Regression model with specified parameters

    training_df - dataframe with preprocessed features
    params - dictionary with hyperparameters
    target_col - name of target column
    isClassifier - boolean flag for classification vs regression
    pipeline_stages - list of preprocessing stages
    weightCol - (optional) name of column containing sample weights
    """
    # Validate that LogisticRegression is only used for clf
    if not isClassifier:
        raise ValueError(
            "LogisticRegression is only applicable for classification tasks. "
            "Set isClassifier=True or use a different algorithm for regression."
        )

    # Determine if binary clf for threshold param
    unique_labels = training_df.select(target_col).distinct().collect()
    is_binary = len(unique_labels) == 2

    # Create LogisticRegression model with hyperpara settings
    lr_params = {
        "featuresCol": "final_features",
        "labelCol": target_col,
        "predictionCol": "prediction",
        "regParam": params["regParam"],
        "elasticNetParam": params["elasticNetParam"],
        "maxIter": params["maxIter"],
        "tol": params["tol"],
        "fitIntercept": params["fitIntercept"],
        "standardization": params["standardization"],
        "aggregationDepth": params["aggregationDepth"],
    }

    # Add weightCol if provided
    if weightCol is not None:
        lr_params["weightCol"] = weightCol

    # Add threshold for binary clf
    if is_binary and params.get("threshold") is not None:
        lr_params["threshold"] = params["threshold"]

    lr = LogisticRegression(**lr_params)

    # Add LogisticRegression to pipeline stages
    pipeline_stages.append(lr)

    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(training_df)

    return full_pipeline


def evaluate_lr_model(
    input_df,
    params,
    full_pipeline,
    target_col,
    isClassifier,
    id_col="record",
    date_col="evaluation_date",
):
    """
    Evaluate a trained Logistic Regression model on input data

    input_df - dataframe to evaluate on
    params - dictionary with hyperparameters used
    full_pipeline - trained pipeline model
    target_col - name of target column
    isClassifier - boolean flag for classification vs regression
    id_col - name of the ID column (default: "record")
    date_col - name of the date column (default: "evaluation_date")
    """
    # Validate that LogisticRegression is used for clf
    if not isClassifier:
        raise ValueError(
            "LogisticRegression is only applicable for classification tasks. "
            "Set isClassifier=True or use a different algorithm for regression."
        )

    performance = []
    pred_df = full_pipeline.transform(input_df)

    # Get number of unique classes in the actual target
    distinct_target_values = input_df.select(target_col).distinct().count()
    isBinary = distinct_target_values == 2

    # Check the predictions to confirm binary vs multiclass
    distinct_predictions = pred_df.select("prediction").distinct().count()

    pred_df = pred_df.withColumn(
        "probability_array", vector_to_array(col("probability"))
    )

    if isBinary:
        # For binary classification
        pred_df = pred_df.withColumn("probability1", col("probability_array")[1])
        pred_df = pred_df.select(
            id_col,
            date_col,
            target_col,
            "prediction",
            "probability1",
            "probability_array",
            "probability",
            "rawPrediction",
        )

        # Only attempt binary metrics if both classes in the predictions and targets
        if distinct_target_values == 2 and distinct_predictions == 2:
            try:
                binary_eval = BinaryClassificationEvaluator(
                    labelCol=target_col,
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderROC",
                )
                auROC = binary_eval.evaluate(pred_df)
                binary_eval.setMetricName("areaUnderPR")
                auPR = binary_eval.evaluate(pred_df)
            except Exception as e:
                print(f"Binary evaluation failed: {e}")
                auROC = None
                auPR = None
        else:
            auROC = None
            auPR = None
    else:
        # For multiclass, no probability1 column and no binary metrics
        pred_df = pred_df.select(
            id_col,
            date_col,
            target_col,
            "prediction",
            "probability_array",
            "probability",
            "rawPrediction",
        )
        # Do not calculate binary metrics for multiclass
        auROC = None
        auPR = None

    # Multiclass metrics (also work for binary)
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator_accuracy.evaluate(pred_df)
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="weightedPrecision",
    )
    precision = evaluator_precision.evaluate(pred_df)
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(pred_df)
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator_f1.evaluate(pred_df)

    # Get class-specific metrics
    class_metrics_df, confusion_matrix_df = evaluate_all_classes(
        pred_df, target_col, "prediction"
    )

    perf_dict = {
        "regParam": params["regParam"],
        "elasticNetParam": params["elasticNetParam"],
        "maxIter": params["maxIter"],
        "tol": params["tol"],
        "fitIntercept": params["fitIntercept"],
        "standardization": params["standardization"],
        "threshold": params.get("threshold"),
        "aggregationDepth": params["aggregationDepth"],
        "auROC": auROC,
        "auPR": auPR,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse": None,
        "mae": None,
        "r2": None,
    }
    performance.append(perf_dict)

    # Define output schema
    schema = StructType(
        [
            StructField("regParam", DoubleType(), True),
            StructField("elasticNetParam", DoubleType(), True),
            StructField("maxIter", IntegerType(), True),
            StructField("tol", DoubleType(), True),
            StructField("fitIntercept", BooleanType(), True),
            StructField("standardization", BooleanType(), True),
            StructField("threshold", DoubleType(), True),
            StructField("aggregationDepth", IntegerType(), True),
            StructField("auROC", DoubleType(), True),
            StructField("auPR", DoubleType(), True),
            StructField("accuracy", DoubleType(), True),
            StructField("precision", DoubleType(), True),
            StructField("recall", DoubleType(), True),
            StructField("f1", DoubleType(), True),
            StructField("rmse", DoubleType(), True),
            StructField("mae", DoubleType(), True),
            StructField("r2", DoubleType(), True),
        ]
    )

    spark = SparkSession.builder.getOrCreate()
    df_performance = spark.createDataFrame(performance, schema=schema)
    return df_performance, pred_df, class_metrics_df, confusion_matrix_df
