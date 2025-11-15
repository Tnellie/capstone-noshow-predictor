import math
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    RandomForestClassifier,  # used to fit the model
    RandomForestClassificationModel,  # fitted model object
)
from pyspark.ml.regression import (
    RandomForestRegressor,  # used to fit the model
    RandomForestRegressionModel,  # fitted model object
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
)
from prismSparkML.modelEvaluation import evaluate_all_classes
from prismSparkML.modelingSupport import random_parameter_search


def CV_RandomForest(
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
    Cross-validation hyperparameter search for Random Forest

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
    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(cv_df)
    validation_df = full_pipeline.transform(cv_df)

    # Split validation and test
    K = validation_df.agg(F.max(F.col(fold_col)).alias("max_value")).collect()[0][
        "max_value"
    ]

    # Define parameter space based on model type
    if isClassifier:
        param_space = {
            "numTrees": [10, 25, 50, 100, 150, 200, 300, 500],  # Number of trees
            "maxDepth": [3, 5, 8, 10, 15, 20, 25, 30],  # Maximum depth of trees
            "minInstancesPerNode": [
                1,
                2,
                5,
                10,
                15,
                20,
                30,
                50,
            ],  # Min instances per node
            "maxBins": [32, 64, 96, 128, 256],  # Max bins for discretizing
            "featureSubsetStrategy": [
                "auto",
                "sqrt",
                "log2",
                "onethird",
            ],  # Features per split
            "subsamplingRate": [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsampling rate
            "impurity": ["gini", "entropy"],  # Impurity measure
        }
    else:
        param_space = {
            "numTrees": [10, 25, 50, 100, 150, 200, 300, 500],  # Number of trees
            "maxDepth": [3, 5, 8, 10, 15, 20, 25, 30],  # Maximum depth of trees
            "minInstancesPerNode": [
                1,
                2,
                5,
                10,
                15,
                20,
                30,
                50,
            ],  # Min instances per node
            "maxBins": [32, 64, 96, 128, 256],  # Max bins for discretizing
            "featureSubsetStrategy": [
                "auto",
                "sqrt",
                "log2",
                "onethird",
            ],  # Features per split
            "subsamplingRate": [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsampling rate
            "impurity": ["variance"],  # Impurity measure for regression
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

    # Use n_param_combos random param value combos or 25% of total, whichever is smaller
    percentage = 0.25 if not isClassifier else 0.1
    num_random_combinations = min(
        n_param_combos, math.ceil(total_combinations * percentage)
    )

    # Use the random parameter search function
    hp_list = random_parameter_search(
        param_space=param_space, num_samples=num_random_combinations, seed=42
    )

    val_performance = []

    for hp_dict in hp_list:
        if isClassifier:
            rf_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "probabilityCol": "probability",
                "rawPredictionCol": "rawPrediction",
                "numTrees": hp_dict["numTrees"],
                "maxDepth": hp_dict["maxDepth"],
                "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                "maxBins": hp_dict["maxBins"],
                "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                "subsamplingRate": hp_dict["subsamplingRate"],
                "impurity": hp_dict["impurity"],
                "seed": 42,
            }
            # Add weight col if provided
            if weightCol is not None:
                rf_params["weightCol"] = weightCol

            rf = RandomForestClassifier(**rf_params)
        else:
            # Regression params
            rf_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "numTrees": hp_dict["numTrees"],
                "maxDepth": hp_dict["maxDepth"],
                "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                "maxBins": hp_dict["maxBins"],
                "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                "subsamplingRate": hp_dict["subsamplingRate"],
                "impurity": hp_dict["impurity"],
                "seed": 42,
            }
            # Add weight col if provided
            if weightCol is not None:
                rf_params["weightCol"] = weightCol

            rf = RandomForestRegressor(**rf_params)

        # Cross-validation loop
        for k in range(1, K + 1):  # Start from 1, not 0
            validation_train = validation_df.filter(
                validation_df[fold_col].isin([i for i in range(1, K + 1) if i != k])
            )
            validation_test = validation_df.filter(validation_df[fold_col] == k)

            rf_model = rf.fit(validation_train)
            pred_df = rf_model.transform(validation_test)

            if isClassifier:
                # Determine if binary classification
                Cnt = validation_df.groupBy(target_col).count()
                isBinary = Cnt.count() == 2

                pred_df = pred_df.select(
                    id_col,
                    date_col,
                    target_col,
                    "prediction",
                    "probability",
                    "rawPrediction",
                )

                if isBinary:
                    # Extract probability for positive class
                    pred_df = pred_df.withColumn(
                        "probability_array", vector_to_array(F.col("probability"))
                    )
                    pred_df = pred_df.withColumn(
                        "probability_positive",
                        F.col("probability_array")[1],  # Positive class probability
                    )

                    # Check for both classes before calculating binary metrics
                    distinct_labels = pred_df.select(target_col).distinct().count()
                    distinct_preds = pred_df.select("prediction").distinct().count()

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
                        print(
                            f"Warning: Fold {k} has insufficient class diversity - skipping AUC metrics"
                        )
                        auROC = float("nan")
                        auPR = float("nan")
                else:
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
                    "numTrees": hp_dict["numTrees"],
                    "maxDepth": hp_dict["maxDepth"],
                    "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                    "maxBins": hp_dict["maxBins"],
                    "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                    "subsamplingRate": hp_dict["subsamplingRate"],
                    "impurity": hp_dict["impurity"],
                    "fold_id": k,
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
                    "numTrees": hp_dict["numTrees"],
                    "maxDepth": hp_dict["maxDepth"],
                    "minInstancesPerNode": hp_dict["minInstancesPerNode"],
                    "maxBins": hp_dict["maxBins"],
                    "featureSubsetStrategy": hp_dict["featureSubsetStrategy"],
                    "subsamplingRate": hp_dict["subsamplingRate"],
                    "impurity": hp_dict["impurity"],
                    "fold_id": k,
                    "weightCol": weightCol,
                    "auROC": float("nan"),
                    "auPR": float("nan"),
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "rmse": rmse if rmse is not None else float("nan"),
                    "mae": mae if mae is not None else float("nan"),
                    "r2": r2 if r2 is not None else float("nan"),
                }

                val_performance.append(hp_performance_dict)

    # Define output schema
    schema = StructType(
        [
            StructField("numTrees", IntegerType(), True),
            StructField("maxDepth", IntegerType(), True),
            StructField("minInstancesPerNode", IntegerType(), True),
            StructField("maxBins", IntegerType(), True),
            StructField("featureSubsetStrategy", StringType(), True),
            StructField("subsamplingRate", DoubleType(), True),
            StructField("impurity", StringType(), True),
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


def train_rf_model(
    training_df, params, target_col, isClassifier, pipeline_stages, weightCol=None
):
    """
    Train a Random Forest model with specified parameters

    training_df - dataframe with preprocessed features
    params - dictionary with hyperparameters
    target_col - name of target column
    isClassifier - boolean flag for classification vs regression
    pipeline_stages - list of preprocessing stages
    weightCol - (optional) name of column containing sample weights
    """

    # Create RF model
    if isClassifier:
        # Build params conditionally to handle weightCol
        rf_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "probabilityCol": "probability",
            "rawPredictionCol": "rawPrediction",
            "numTrees": params["numTrees"],
            "maxDepth": params["maxDepth"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "maxBins": params["maxBins"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "subsamplingRate": params["subsamplingRate"],
            "impurity": params["impurity"],
            "seed": 42,
        }
        # Only add weightCol if not None
        if params.get("weightCol") is not None:
            rf_params["weightCol"] = params["weightCol"]

        rf = RandomForestClassifier(**rf_params)
    else:
        # Build params conditionally to handle weightCol properly
        rf_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "numTrees": params["numTrees"],
            "maxDepth": params["maxDepth"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "maxBins": params["maxBins"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "subsamplingRate": params["subsamplingRate"],
            "impurity": params["impurity"],
            "seed": 42,
        }
        # Only add weightCol if not None
        if params.get("weightCol") is not None:
            rf_params["weightCol"] = params["weightCol"]

        rf = RandomForestRegressor(**rf_params)

    pipeline_stages.append(rf)

    # Apply pipeline to DF
    pipeline = Pipeline(stages=pipeline_stages)
    full_pipeline = pipeline.fit(training_df)

    return full_pipeline


def evaluate_rf_model(
    input_df,
    params,
    full_pipeline,
    target_col,
    isClassifier,
    id_col="record",
    date_col="evaluation_date",
):
    """
    Evaluate a trained Random Forest model on input data.

    This function works with all pipelines but provides enhanced feature importance
    interpretation when using pipelines that contain a feature map dictionary. For optimal
    feature importance results, create your pipeline using the create_feature_pipeline()
    function from prismSparkML.modelingSupport, which automatically includes a feature map.

    Without a feature map, the function falls back to using generic feature names
    (feature_0, feature_1, etc.) in the feature importance results.

    Parameters:
    -----------
    input_df : DataFrame
        Dataframe to evaluate the model on
    params : dict
        Dictionary with hyperparameters used for the Random Forest model
    full_pipeline : PipelineModel
        Trained pipeline model (preferably created with create_feature_pipeline())
    target_col : str
        Name of target column in the dataframe
    isClassifier : bool
        Flag indicating if this is a classification model (True) or regression model (False)
    id_col : str, default="record"
        Name of the ID column in the dataframe
    date_col : str, default="evaluation_date"
        Name of the date column in the dataframe

    Returns:
    --------
    tuple
        A tuple containing:
        - df_performance: DataFrame with model performance metrics
        - pred_df: DataFrame with model predictions
        - class_metrics_df: DataFrame with class-specific metrics (classification only)
        - confusion_matrix_df: DataFrame with confusion matrix (classification only)
        - feature_importance_df: DataFrame with feature importance values

    Notes:
    ------
    To ensure detailed feature importance interpretation, create your pipeline using:
    from prismSparkML.modelingSupport import create_feature_pipeline
    """
    performance = []
    pred_df = full_pipeline.transform(input_df)

    # Extract feature importance from RF model
    rf_model = None
    for stage in full_pipeline.stages:
        if isinstance(stage, RandomForestClassificationModel) or isinstance(
            stage, RandomForestRegressionModel
        ):
            rf_model = stage
            break

    # Extract the feature map from the pipeline
    feature_map = None
    for stage in full_pipeline.stages:
        if hasattr(stage, "getFeatureMap"):
            feature_map = stage.getFeatureMap()
            break

    feature_importance_df = None

    if rf_model is not None:
        # Get feature importance from the model
        feature_importances = rf_model.featureImportances

        # Convert to array if needed
        if hasattr(feature_importances, "toArray"):
            feature_importances = feature_importances.toArray()

        spark = SparkSession.builder.getOrCreate()

        # Use the feature map if available; fallback to indices
        if feature_map:
            feature_records = []

            for i, importance in enumerate(feature_importances):
                # Get feature name from map or use index as fallback
                feature_name = feature_map.get(i, f"feature_{i}")
                feature_records.append((feature_name, float(importance)))

            feature_importance_df = spark.createDataFrame(
                feature_records, ["feature", "importance"]
            )

            # Sort by importance (descending)
            feature_importance_df = feature_importance_df.orderBy(
                F.col("importance").desc()
            )
        else:
            # Fallback to using indices
            feature_importance_df = spark.createDataFrame(
                [
                    (f"feature_{i}", float(imp))
                    for i, imp in enumerate(feature_importances)
                ],
                ["feature", "importance"],
            )

    if isClassifier:
        # Determine if binary classification
        Cnt = input_df.groupBy(target_col).count()
        isBinary = Cnt.count() == 2

        # Extract probability array for all classification cases
        pred_df = pred_df.withColumn(
            "probability_array", vector_to_array(F.col("probability"))
        )

        if isBinary:
            # Extract probability for positive class
            pred_df = pred_df.withColumn(
                "probability_positive",
                F.col("probability_array")[1],  # Positive class probability
            )

            # Select cols including probability_positive for binary
            pred_df = pred_df.select(
                id_col,
                date_col,
                target_col,
                "prediction",
                "probability_positive",
                "probability_array",
                "probability",
                "rawPrediction",
            )

            # Check for both classes before calculating binary metrics
            distinct_labels = pred_df.select(target_col).distinct().count()
            distinct_preds = pred_df.select("prediction").distinct().count()

            if distinct_labels >= 2 and distinct_preds >= 2:
                try:
                    # Binary classification metrics using rawPrediction vector
                    binary_eval = BinaryClassificationEvaluator(
                        labelCol=target_col,
                        rawPredictionCol="rawPrediction",
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
                auROC = None
                auPR = None
        else:
            # For multiclass, no probability_positive column
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

        # Create performance dict with RF hyperparams
        perf_dict = {
            "numTrees": params["numTrees"],
            "maxDepth": params["maxDepth"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "maxBins": params["maxBins"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "subsamplingRate": params["subsamplingRate"],
            "impurity": params["impurity"],
            "weightCol": params.get("weightCol"),
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

        # Create performance dict with RF hyperparams
        perf_dict = {
            "numTrees": params["numTrees"],
            "maxDepth": params["maxDepth"],
            "minInstancesPerNode": params["minInstancesPerNode"],
            "maxBins": params["maxBins"],
            "featureSubsetStrategy": params["featureSubsetStrategy"],
            "subsamplingRate": params["subsamplingRate"],
            "impurity": params["impurity"],
            "weightCol": params.get("weightCol"),
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
            StructField("numTrees", IntegerType(), True),
            StructField("maxDepth", IntegerType(), True),
            StructField("minInstancesPerNode", IntegerType(), True),
            StructField("maxBins", IntegerType(), True),
            StructField("featureSubsetStrategy", StringType(), True),
            StructField("subsamplingRate", DoubleType(), True),
            StructField("impurity", StringType(), True),
            StructField("weightCol", StringType(), True),
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
    return (
        df_performance,
        pred_df,
        class_metrics_df,
        confusion_matrix_df,
        feature_importance_df,
    )
