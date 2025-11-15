import math
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.regression import LinearRegression
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
    BooleanType,
)
from prismSparkML.modelEvaluation import evaluate_all_classes
from prismSparkML.modelingSupport import random_parameter_search


def CV_SupportVectorMachine(
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
    This script does a cross-validation hyperparameter search through various Support Vector Machine Settings

    Note: For classification, this only supports BINARY classification (labels must be 0 and 1).
    LinearSVC does not support multiclass classification or probability estimates.

    cv_df - dataframe with:
    id_col - The record ID (default: "record")
    date_col - the date associated with the data and record (default: "evaluation_date")
    target_col - the column with the ML target
    fold_col - a column with 0-indexed folds for training data (default: "fold_id")
    "final_features" - input feature vector processed by a pipeline

    isClassifier - boolean flag: True for binary classification, False for regression
    weightCol - (optional) name of the column containing instance weights for training
    fold_col - name of the fold column (default: "fold_id")
    id_col - name of the ID column (default: "record")
    date_col - name of the date column (default: "evaluation_date")
    n_param_combos - (int) number of random hyperparameter combinations to test during cross-validation
    param_overrides - (optional) dict to override specific parameter values
            Example: {"solver": ["l-bfgs"], "regParam": [0.01, 0.1]}
    """
    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(cv_df)
    validation_df = full_pipeline.transform(cv_df)

    # Get number of folds
    K = validation_df.agg(F.max(F.col(fold_col)).alias("max_value")).collect()[0][
        "max_value"
    ]

    # Validate binary clf is classifier
    if isClassifier:
        unique_labels = validation_df.select(target_col).distinct().collect()
        label_values = sorted([row[0] for row in unique_labels])
        if label_values != [0, 1]:
            raise ValueError(
                f"LinearSVC only supports binary classification with labels [0, 1]. Found labels: {label_values}"
            )

    # Define parameter spaces
    if isClassifier:
        # LinearSVC hyperparameters
        param_space = {
            "regParam": [
                0.001,
                0.01,
                0.1,
                0.5,
                1.0,
                5.0,
                10.0,
                50.0,
                100.0,
                500.0,
            ],  # Regularization parameter (C inverse)
            "maxIter": [10, 50, 100, 200, 500, 1000, 2000],  # Max iterations
            "tol": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],  # Convergence tolerance
            "fitIntercept": [True, False],  # Whether to fit intercept
            "standardization": [True, False],  # Whether to standardize features
            "threshold": [
                0.0,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],  # Decision threshold (0.0 is default)
            "aggregationDepth": [2, 4, 8],  # Aggregation depth for tree reduce
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

        # Use n_param_combos random param value combos or 45% of total, whichever is smaller
        num_random_combinations = min(
            n_param_combos, math.ceil(total_combinations * 0.45)
        )
    else:
        # LinearRegression hyperparameters (using as SVM regression alternative)
        param_space = {
            "regParam": [
                0.0,  # No regularization
                0.00001,  # Very weak
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,  # Weak
                0.05,
                0.1,  # Moderate
                0.5,
                1.0,
                2.0,
                5.0,  # Strong
                10.0,
                25.0,
                50.0,  # Very strong
                100.0,
                500.0,
                1000,
            ],  # L2 regularization
            "elasticNetParam": [
                0.0,  # Pure L2 (Ridge)
                0.1,  # L2 dominant
                0.2,
                0.3,
                0.4,
                0.5,  # Balanced
                0.6,
                0.7,
                0.8,  # L1 dominant
                0.9,
                1.0,  # Pure L1 (Lasso)
            ],
            "maxIter": [10, 25, 50, 100, 200, 500, 1000, 2000],  # Max iterations
            "tol": [
                1e-8,  # Very strict
                1e-7,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,  # Relaxed
            ],  # Convergence tolerance
            "fitIntercept": [True, False],  # Whether to fit intercept
            "standardization": [True, False],  # Whether to standardize features
            "solver": ["auto", "normal", "l-bfgs"],  # Solver algorithm
            "aggregationDepth": [2, 3, 4],  # Aggregation depth for tree reduce
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

        # Calculate total combos
        total_combinations = 1
        for values in param_space.values():
            total_combinations *= len(values)

        # Use n_param_combos random param value combos or 10% of total, whichever is smaller
        num_random_combinations = min(
            n_param_combos, math.ceil(total_combinations * 0.1)
        )

    # Call random parameter search function
    hp_list = random_parameter_search(
        param_space=param_space, num_samples=num_random_combinations, seed=42
    )

    val_performance = []

    for hp_dict in hp_list:
        if isClassifier:
            svm_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "rawPredictionCol": "rawPrediction",
                "regParam": hp_dict["regParam"],
                "maxIter": hp_dict["maxIter"],
                "tol": hp_dict["tol"],
                "fitIntercept": hp_dict["fitIntercept"],
                "standardization": hp_dict["standardization"],
                "threshold": hp_dict["threshold"],
                "aggregationDepth": hp_dict["aggregationDepth"],
            }
            # Add weight column if provided
            if weightCol is not None:
                svm_params["weightCol"] = weightCol

            svm = LinearSVC(**svm_params)
        else:
            # Regression params
            svm_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "regParam": hp_dict["regParam"],
                "elasticNetParam": hp_dict["elasticNetParam"],
                "maxIter": hp_dict["maxIter"],
                "tol": hp_dict["tol"],
                "fitIntercept": hp_dict["fitIntercept"],
                "standardization": hp_dict["standardization"],
                "solver": hp_dict["solver"],
                "aggregationDepth": hp_dict["aggregationDepth"],
            }
            # Add weight column if provided
            if weightCol is not None:
                svm_params["weightCol"] = weightCol

            svm = LinearRegression(**svm_params)

        # Cross-validation loop
        for k in range(1, K + 1):  # Start from 1, not 0
            validation_train = validation_df.filter(
                validation_df[fold_col].isin([i for i in range(1, K + 1) if i != k])
            )
            validation_test = validation_df.filter(validation_df[fold_col] == k)

            # Skip empty folds
            if validation_test.rdd.isEmpty():
                continue

            svm_model = svm.fit(validation_train)
            pred_df = svm_model.transform(validation_test)

            if isClassifier:
                # Binary clf only
                pred_df = pred_df.select(
                    id_col,
                    date_col,
                    target_col,
                    "prediction",
                    "rawPrediction",  # Keep rawPrediction for AUC metrics
                )

                # Check for both classes in the fold
                distinct_labels = pred_df.select(target_col).distinct().count()
                distinct_preds = pred_df.select("prediction").distinct().count()

                # Calculate binary metrics only if both classes exist
                if distinct_labels >= 2 and distinct_preds >= 2:
                    try:
                        # Convert rawPrediction vector to array and extract positive class score (index 1)
                        pred_df = pred_df.withColumn(
                            "rawPrediction_array",
                            vector_to_array(F.col("rawPrediction")),
                        )
                        pred_df = pred_df.withColumn(
                            "decision_value",
                            F.col("rawPrediction_array")[1],  # Positive class score
                        )

                        # Binary clf metrics using decision values
                        binary_eval = BinaryClassificationEvaluator(
                            labelCol=target_col,
                            rawPredictionCol="decision_value",
                            metricName="areaUnderROC",
                        )
                        auROC = binary_eval.evaluate(pred_df)
                        binary_eval.setMetricName("areaUnderPR")
                        auPR = binary_eval.evaluate(pred_df)
                    except Exception as e:
                        print(f"Warning: Could not calculate AUC metrics: {e}")
                        auROC = float("nan")
                        auPR = float("nan")
                else:
                    # Skip binary metrics if only one class present
                    print(
                        f"Warning: Fold {k} has insufficient class diversity - skipping AUC metrics"
                    )
                    auROC = float("nan")
                    auPR = float("nan")

                # Multiclass metrics (also work for binary)
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

                # Create hyperparam dict for classification
                hp_performance_dict = {
                    "regParam": hp_dict["regParam"],
                    "maxIter": hp_dict["maxIter"],
                    "tol": hp_dict["tol"],
                    "fitIntercept": hp_dict["fitIntercept"],
                    "standardization": hp_dict["standardization"],
                    "threshold": hp_dict["threshold"],
                    "aggregationDepth": hp_dict["aggregationDepth"],
                    "fold": k,
                    "weightCol": weightCol,
                    "auROC": auROC if auROC is not None else float("nan"),
                    "auPR": auPR if auPR is not None else float("nan"),
                    "accuracy": accuracy if accuracy is not None else float("nan"),
                    "precision": precision if precision is not None else float("nan"),
                    "recall": recall if recall is not None else float("nan"),
                    "f1": f1 if f1 is not None else float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "elasticNetParam": float("nan"),
                    "solver": None,
                }

                val_performance.append(hp_performance_dict)
            else:
                # Regression metrics
                pred_df = pred_df.select(id_col, date_col, target_col, "prediction")

                rmse_evaluator = RegressionEvaluator(
                    labelCol=target_col, predictionCol="prediction", metricName="rmse"
                )
                rmse = rmse_evaluator.evaluate(pred_df)

                mae_evaluator = RegressionEvaluator(
                    labelCol=target_col, predictionCol="prediction", metricName="mae"
                )
                mae = mae_evaluator.evaluate(pred_df)

                r2_evaluator = RegressionEvaluator(
                    labelCol=target_col, predictionCol="prediction", metricName="r2"
                )
                r2 = r2_evaluator.evaluate(pred_df)

                # Create hyperparam dict for regression
                hp_performance_dict = {
                    "regParam": hp_dict["regParam"],
                    "elasticNetParam": hp_dict["elasticNetParam"],
                    "maxIter": hp_dict["maxIter"],
                    "tol": hp_dict["tol"],
                    "fitIntercept": hp_dict["fitIntercept"],
                    "standardization": hp_dict["standardization"],
                    "solver": hp_dict["solver"],
                    "aggregationDepth": hp_dict["aggregationDepth"],
                    "fold": k,
                    "weightCol": weightCol,
                    "auROC": float("nan"),
                    "auPR": float("nan"),
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    # Regression metrics
                    "rmse": rmse if rmse is not None else float("nan"),
                    "mae": mae if mae is not None else float("nan"),
                    "r2": r2 if r2 is not None else float("nan"),
                    "threshold": float("nan"),
                }

                val_performance.append(hp_performance_dict)

    # Define output schema
    schema = StructType(
        [
            StructField("regParam", DoubleType(), True),
            StructField("elasticNetParam", DoubleType(), True),
            StructField("maxIter", IntegerType(), True),
            StructField("tol", DoubleType(), True),
            StructField("fitIntercept", BooleanType(), True),
            StructField("standardization", BooleanType(), True),
            StructField("solver", StringType(), True),
            StructField("threshold", DoubleType(), True),
            StructField("aggregationDepth", IntegerType(), True),
            StructField("weightCol", StringType(), True),
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

    spark = SparkSession.builder.getOrCreate()
    df_performance = spark.createDataFrame(val_performance, schema=schema)
    return df_performance


def train_svm_model(
    training_df, params, target_col, isClassifier, pipeline_stages, weightCol=None
):
    """
    Train a Support Vector Machine model with specified parameters
    """

    # Create SVM model
    if isClassifier:
        svm_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "rawPredictionCol": "rawPrediction",
            "regParam": params["regParam"],
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "fitIntercept": params["fitIntercept"],
            "standardization": params["standardization"],
            "threshold": params["threshold"],
            "aggregationDepth": params["aggregationDepth"],
        }
        # Use the weightCol parameter
        if weightCol is not None:
            svm_params["weightCol"] = weightCol

        svm = LinearSVC(**svm_params)
    else:
        svm_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "regParam": params["regParam"],
            "elasticNetParam": params["elasticNetParam"],
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "fitIntercept": params["fitIntercept"],
            "standardization": params["standardization"],
            "solver": params["solver"],
            "aggregationDepth": params["aggregationDepth"],
        }
        # Use the weightCol parameter
        if weightCol is not None:
            svm_params["weightCol"] = weightCol

        svm = LinearRegression(**svm_params)

    pipeline_stages.append(svm)

    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(training_df)

    return full_pipeline


def evaluate_svm_model(
    input_df,
    params,
    full_pipeline,
    target_col,
    isClassifier,
    id_col="record",
    date_col="evaluation_date",
):
    """
    Evaluate a trained SVM model on input data

    input_df - dataframe to evaluate on
    params - dictionary with hyperparameters used
    full_pipeline - trained pipeline model
    target_col - name of target column
    isClassifier - boolean flag for classification vs regression
    id_col - name of the ID column (default: "record")
    date_col - name of the date column (default: "evaluation_date")
    """
    performance = []
    pred_df = full_pipeline.transform(input_df)

    if isClassifier:
        # Determine if binary classification
        Cnt = input_df.groupBy(target_col).count()
        isBinary = Cnt.count() == 2

        # Select relevant columns
        pred_df = pred_df.select(
            id_col,
            date_col,
            target_col,
            "prediction",
            "rawPrediction",
        )

        # Check for both classes before calculating binary metrics
        distinct_labels = pred_df.select(target_col).distinct().count()
        distinct_preds = pred_df.select("prediction").distinct().count()

        if isBinary and distinct_labels >= 2 and distinct_preds >= 2:
            try:
                # Extract decision value for ROC calculation
                pred_df = pred_df.withColumn(
                    "rawPrediction_array", vector_to_array(F.col("rawPrediction"))
                )
                pred_df = pred_df.withColumn(
                    "decision_value",
                    F.col("rawPrediction_array")[1],  # Positive class score
                )

                # Binary classification metrics using decision values
                binary_eval = BinaryClassificationEvaluator(
                    labelCol=target_col,
                    rawPredictionCol="decision_value",
                    metricName="areaUnderROC",
                )
                auROC = binary_eval.evaluate(pred_df)
                binary_eval.setMetricName("areaUnderPR")
                auPR = binary_eval.evaluate(pred_df)
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
                auROC = None
                auPR = None
        else:
            print("Warning: Insufficient class diversity - skipping AUC metrics")
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

        # Create performance dict with SVM hyperparams
        perf_dict = {
            "regParam": params["regParam"],
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "fitIntercept": params["fitIntercept"],
            "standardization": params["standardization"],
            "threshold": params["threshold"],
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
            "elasticNetParam": None,
            "solver": None,
        }
        performance.append(perf_dict)

    else:
        # Regression evaluation
        pred_df = pred_df.select(id_col, date_col, target_col, "prediction")

        rmse_evaluator = RegressionEvaluator(
            labelCol=target_col, predictionCol="prediction", metricName="rmse"
        )
        rmse = rmse_evaluator.evaluate(pred_df)

        mae_evaluator = RegressionEvaluator(
            labelCol=target_col, predictionCol="prediction", metricName="mae"
        )
        mae = mae_evaluator.evaluate(pred_df)

        r2_evaluator = RegressionEvaluator(
            labelCol=target_col, predictionCol="prediction", metricName="r2"
        )
        r2 = r2_evaluator.evaluate(pred_df)

        # For regression, set class metrics to None
        class_metrics_df = None
        confusion_matrix_df = None

        # Create performance dict with SVM hyperparameters
        perf_dict = {
            "regParam": params["regParam"],
            "elasticNetParam": params["elasticNetParam"],
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "fitIntercept": params["fitIntercept"],
            "standardization": params["standardization"],
            "solver": params["solver"],
            "aggregationDepth": params["aggregationDepth"],
            "auROC": None,
            "auPR": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "threshold": None,
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
            StructField("solver", StringType(), True),
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
