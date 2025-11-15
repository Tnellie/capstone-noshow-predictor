import math
from pyspark.sql import functions as F
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
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
)
from prismSparkML.modelEvaluation import evaluate_all_classes
from prismSparkML.modelingSupport import random_parameter_search


def CV_GradientBoostedTree(
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
    This script does a cross-validation hyperparameter search through various Gradient Boosted Tree Settings

    Note: For classification, this only supports BINARY classification (labels must be 0 and 1).
    GBTClassifier does not support multiclass classification.

    cv_df - dataframe with:
    id_col - The record ID (default: "record")
    date_col - the date associated with the data and record (default: "evaluation_date")
    target_col - the column with the ML target
    "is_test" - A column equal to 1 if test and 0 if training
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

    # Apply pipeline to dataframe
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(cv_df)
    validation_df = full_pipeline.transform(cv_df)

    # Get number of folds
    K = validation_df.agg(F.max(F.col(fold_col)).alias("max_value")).collect()[0][
        "max_value"
    ]

    # Validate binary classification if classifier
    if isClassifier:
        unique_labels = validation_df.select(target_col).distinct().collect()
        label_values = sorted([row[0] for row in unique_labels])
        if label_values != [0, 1]:
            raise ValueError(
                f"GBTClassifier only supports binary classification with labels [0, 1]. Found labels: {label_values}"
            )

    # Define parameter space for GBT
    param_space = {
        "maxIter": [10, 20, 50, 100, 200, 300, 500],  # Number of boosting iterations
        "maxDepth": [2, 3, 4, 5, 7, 10],  # Max depth of trees
        "stepSize": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],  # Learning rate
        "subsamplingRate": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Subsampling rate
        "featureSubsetStrategy": [
            "auto",
            "onethird",
            "sqrt",
            "log2",
        ],  # Feature subset strategy
        "minInstancesPerNode": [
            1,
            2,
            5,
            10,
            20,
            30,
        ],  # Min number of instances required at a leaf node
        "minInfoGain": [
            0.0,
            0.01,
            0.05,
            0.1,
        ],  # Min reduction in impurity required to split a node
        "maxBins": [
            16,
            32,
            64,
            128,
            256,
        ],  # Number of bins used for discretizing continuous features
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

    # Use n_param_combos random param value combos or 20% of total, whichever is smaller
    num_random_combinations = min(n_param_combos, math.ceil(total_combinations * 0.20))

    # Use the random parameter search function
    hp_list = random_parameter_search(
        param_space=param_space, num_samples=num_random_combinations, seed=42
    )

    val_performance = []

    for hp_dict in hp_list:
        if isClassifier:
            # Build GBT classifier params
            gbt_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "maxIter": hp_dict["maxIter"],
                "maxDepth": hp_dict["maxDepth"],
                "stepSize": hp_dict["stepSize"],
                "subsamplingRate": hp_dict["subsamplingRate"],
                "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                "minInfoGain": hp_dict["minInfoGain"],
                "maxBins": hp_dict["maxBins"],
                "seed": 42,
            }
            # Add weight column if provided
            if weightCol is not None:
                gbt_params["weightCol"] = weightCol

            gbt = GBTClassifier(**gbt_params)
        else:
            # Build GBT regressor params
            gbt_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "maxIter": hp_dict["maxIter"],
                "maxDepth": hp_dict["maxDepth"],
                "stepSize": hp_dict["stepSize"],
                "subsamplingRate": hp_dict["subsamplingRate"],
                "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                "minInfoGain": hp_dict["minInfoGain"],
                "maxBins": hp_dict["maxBins"],
                "seed": 42,
            }
            # Add weight column if provided
            if weightCol is not None:
                gbt_params["weightCol"] = weightCol

            gbt = GBTRegressor(**gbt_params)

        # Cross-validation loop
        for k in range(1, K + 1):  # Start from 1, not 0
            validation_train = validation_df.filter(
                validation_df[fold_col].isin([i for i in range(1, K + 1) if i != k])
            )
            validation_test = validation_df.filter(validation_df[fold_col] == k)

            gbt_model = gbt.fit(validation_train)
            pred_df = gbt_model.transform(validation_test)

            if isClassifier:
                # Add probability cols for classification
                pred_df = pred_df.withColumn(
                    "probability_array", vector_to_array(F.col("probability"))
                )
                pred_df = pred_df.withColumn(
                    "probability1", F.col("probability_array")[1]
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

                # Check for both classes in the fold
                distinct_labels = pred_df.select(target_col).distinct().count()
                distinct_preds = pred_df.select("prediction").distinct().count()

                # Calculate binary metrics only if both classes exist
                if distinct_labels >= 2 and distinct_preds >= 2:
                    # Binary classification metrics
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

                # Multiclass metrics (these should still work even with one class)
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
                        "maxIter": hp_dict["maxIter"],
                        "maxDepth": hp_dict["maxDepth"],
                        "stepSize": hp_dict["stepSize"],
                        "subsamplingRate": hp_dict["subsamplingRate"],
                        "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                        "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                        "minInfoGain": hp_dict["minInfoGain"],
                        "maxBins": hp_dict["maxBins"],
                        "seed": 42,
                        "weightCol": weightCol,
                        "fold_id": k,
                        "auROC": auROC if auROC is not None else float("nan"),
                        "auPR": auPR if auPR is not None else float("nan"),
                        "accuracy": accuracy if accuracy is not None else float("nan"),
                        "precision": precision
                        if precision is not None
                        else float("nan"),
                        "recall": recall if recall is not None else float("nan"),
                        "f1": f1 if f1 is not None else float("nan"),
                        "rmse": float("nan"),
                        "mae": float("nan"),
                        "r2": float("nan"),
                    }
                )
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

                val_performance.append(
                    {
                        "maxIter": hp_dict["maxIter"],
                        "maxDepth": hp_dict["maxDepth"],
                        "stepSize": hp_dict["stepSize"],
                        "subsamplingRate": hp_dict["subsamplingRate"],
                        "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                        "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                        "minInfoGain": hp_dict["minInfoGain"],
                        "maxBins": hp_dict["maxBins"],
                        "seed": 42,
                        "weightCol": weightCol,
                        "fold_id": k,
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
                    }
                )

    # Define output schema
    schema = StructType(
        [
            StructField("maxIter", IntegerType(), True),
            StructField("maxDepth", IntegerType(), True),
            StructField("stepSize", DoubleType(), True),
            StructField("subsamplingRate", DoubleType(), True),
            StructField("featureSubsetStrategy", StringType(), True),
            StructField("minInstancesPerNode", IntegerType(), True),
            StructField("minInfoGain", DoubleType(), True),
            StructField("maxBins", IntegerType(), True),
            StructField("seed", IntegerType(), True),
            StructField("weightCol", StringType(), True),
            StructField("fold_id", IntegerType(), True),
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


def train_gbt_model(
    training_df, params, target_col, isClassifier, pipeline_stages, weightCol=None
):
    """
    Train a Gradient Boosted Tree model with specified parameters

    training_df - dataframe with preprocessed features
    params - dictionary with hyperparameters
    target_col - name of target column
    isClassifier - boolean flag for classification vs regression
    pipeline_stages - list of preprocessing stages
    weightCol - (optional) name of column containing sample weights
    """

    # Fit gradient-boosted tree
    if isClassifier:
        # Create base params without weightCol
        gbt_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "maxIter": params["maxIter"],
            "maxDepth": params["maxDepth"],
            "stepSize": params["stepSize"],
            "subsamplingRate": params["subsamplingRate"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "minInfoGain": params["minInfoGain"],
            "maxBins": params["maxBins"],
            "seed": params["seed"],
        }

        # Only add weightCol if it's not None
        if weightCol is not None:
            gbt_params["weightCol"] = weightCol

        gbt = GBTClassifier(**gbt_params)
    else:
        # Similar approach for regression
        gbt_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "maxIter": params["maxIter"],
            "maxDepth": params["maxDepth"],
            "stepSize": params["stepSize"],
            "subsamplingRate": params["subsamplingRate"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "minInfoGain": params["minInfoGain"],
            "maxBins": params["maxBins"],
            "seed": params["seed"],
        }

        # Only add weightCol if it's not None
        if weightCol is not None:
            gbt_params["weightCol"] = weightCol

        gbt = GBTRegressor(**gbt_params)

    pipeline_stages.append(gbt)

    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(training_df)

    return full_pipeline


def evaluate_gbt_model(
    input_df,
    params,
    full_pipeline,
    target_col,
    isClassifier,
    id_col="record",
    date_col="evaluation_date",
):
    """
    Evaluate a trained GBT model on input data

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
        # Binary classification data check
        Cnt = input_df.groupBy(target_col).count()
        isBinary = Cnt.count() == 2

        # Extract probability array for all classification cases
        pred_df = pred_df.withColumn(
            "probability_array", vector_to_array(F.col("probability"))
        )

        if isBinary:
            # Extract probability for positive class
            pred_df = pred_df.withColumn("probability1", F.col("probability_array")[1])
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
                binary_eval = BinaryClassificationEvaluator(
                    labelCol=target_col,
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderROC",
                )
                auROC = binary_eval.evaluate(pred_df)
                binary_eval.setMetricName("areaUnderPR")
                auPR = binary_eval.evaluate(pred_df)
            else:
                auROC = None
                auPR = None
        else:
            # For multiclass, no probability1 column
            pred_df = pred_df.select(
                id_col,
                date_col,
                target_col,
                "prediction",
                "probability_array",
                "probability",
                "rawPrediction",
            )
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
            "maxIter": params["maxIter"],
            "maxDepth": params["maxDepth"],
            "stepSize": params["stepSize"],
            "subsamplingRate": params["subsamplingRate"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "minInfoGain": params["minInfoGain"],
            "maxBins": params["maxBins"],
            "seed": params["seed"],
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

        # No class-specific metrics for regression
        class_metrics_df = None
        confusion_matrix_df = None

        perf_dict = {
            "maxIter": params["maxIter"],
            "maxDepth": params["maxDepth"],
            "stepSize": params["stepSize"],
            "subsamplingRate": params["subsamplingRate"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "minInfoGain": params["minInfoGain"],
            "maxBins": params["maxBins"],
            "seed": params["seed"],
            "auROC": None,
            "auPR": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
        performance.append(perf_dict)

    # Define output schema
    schema = StructType(
        [
            StructField("maxIter", IntegerType(), True),
            StructField("maxDepth", IntegerType(), True),
            StructField("stepSize", DoubleType(), True),
            StructField("subsamplingRate", DoubleType(), True),
            StructField("featureSubsetStrategy", StringType(), True),
            StructField("minInstancesPerNode", IntegerType(), True),
            StructField("minInfoGain", DoubleType(), True),
            StructField("maxBins", IntegerType(), True),
            StructField("seed", IntegerType(), True),
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
