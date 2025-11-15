import math
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.regression import GeneralizedLinearRegression
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
    ArrayType,
)
from prismSparkML.modelEvaluation import evaluate_all_classes
from prismSparkML.modelingSupport import random_parameter_search


def CV_MultilayerPerceptron(
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
    Cross-validation hyperparameter search for Multilayer Perceptron

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

    # Get number of folds
    K = validation_df.agg(F.max(F.col(fold_col)).alias("max_value")).collect()[0][
        "max_value"
    ]

    # Get input feature size
    sample_features = validation_df.select("final_features").first()[0]
    input_size = len(sample_features)

    if isClassifier:
        # Get number of classes
        unique_labels = validation_df.select(target_col).distinct().collect()
        num_classes = len(unique_labels)
        is_binary = num_classes == 2

        # Define parameter space for MLP classifier
        hidden_layer_options = [
            [16],
            [32],
            [64],
            [128],
            [16, 8],
            [32, 16],
            [64, 32],
            [128, 64],
            [64, 32, 16],
            [128, 64, 32],
            [256, 128, 64],
        ]

        param_space = {
            "hidden_layers": hidden_layer_options,
            "maxIter": [50, 100, 200, 300, 500],
            "stepSize": [0.001, 0.01, 0.03, 0.05, 0.1, 0.3],
            "tol": [1e-6, 1e-5, 1e-4, 1e-3],
            "blockSize": [64, 128, 256],
            "solver": ["l-bfgs", "gd"],
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

        # Use n_param_combos random param value combos or 30% of total, whichever is smaller
        num_random_combinations = min(
            n_param_combos, math.ceil(total_combinations * 0.3)
        )

    else:
        # For regression, use GLM as MLP alternative
        param_space = {
            "family": ["gaussian", "gamma", "poisson", "tweedie"],
            "maxIter": [10, 25, 50, 100, 200],
            "tol": [1e-6, 1e-5, 1e-4, 1e-3],
            "regParam": [0.0, 0.01, 0.1, 0.5, 1.0, 5.0],
            "fitIntercept": [True, False],
            "solver": ["irls", "auto"],
            "variancePower": [0.0, 1.0, 2.0],
            "linkPower": [0.0, 0.5, 1.0],
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

        # Use n_param_combos random param value combos or 20% of total, whichever is smaller
        num_random_combinations = min(
            n_param_combos, math.ceil(total_combinations * 0.2)
        )

    # Use the random parameter search function
    hp_list = random_parameter_search(
        param_space=param_space, num_samples=num_random_combinations, seed=42
    )

    val_performance = []

    for hp_dict in hp_list:
        if isClassifier:
            # Build full layer specification: input -> hidden -> output
            hidden_layers = hp_dict["hidden_layers"]
            full_layers = [input_size] + list(hidden_layers) + [num_classes]

            mlp_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "layers": full_layers,
                "maxIter": hp_dict["maxIter"],
                "stepSize": hp_dict["stepSize"],
                "tol": hp_dict["tol"],
                "blockSize": hp_dict["blockSize"],
                "solver": hp_dict["solver"],
                "seed": 42,
            }
            # Note: MultilayerPerceptronClassifier does not support weightCol

            mlp = MultilayerPerceptronClassifier(**mlp_params)
        else:
            glm_params = {
                "featuresCol": "final_features",
                "labelCol": target_col,
                "predictionCol": "prediction",
                "family": hp_dict["family"],
                "maxIter": hp_dict["maxIter"],
                "tol": hp_dict["tol"],
                "regParam": hp_dict["regParam"],
                "fitIntercept": hp_dict["fitIntercept"],
                "solver": hp_dict["solver"],
                "seed": 42,
            }

            # Add variancePower and linkPower for tweedie family
            if hp_dict["family"] == "tweedie":
                glm_params["variancePower"] = hp_dict["variancePower"]
                glm_params["linkPower"] = hp_dict["linkPower"]

            if weightCol is not None:
                glm_params["weightCol"] = weightCol

            mlp = GeneralizedLinearRegression(**glm_params)

        # Cross-validation loop
        for k in range(1, K + 1):  # Start from 1, not 0
            validation_train = validation_df.filter(
                validation_df[fold_col].isin([i for i in range(1, K + 1) if i != k])
            )
            validation_test = validation_df.filter(validation_df[fold_col] == k)

            # Skip empty folds
            if validation_test.rdd.isEmpty():
                continue

            mlp_model = mlp.fit(validation_train)
            pred_df = mlp_model.transform(validation_test)

            if isClassifier:
                pred_df = pred_df.select(
                    id_col,
                    date_col,
                    target_col,
                    "prediction",
                    "probability",
                )

                if is_binary:
                    # Check for both classes in the fold
                    distinct_labels = pred_df.select(target_col).distinct().count()
                    distinct_preds = pred_df.select("prediction").distinct().count()

                    # Calculate binary metrics only if both classes exist
                    if distinct_labels >= 2 and distinct_preds >= 2:
                        try:
                            # Extract probability for positive class
                            extract_prob = F.udf(lambda v: float(v[1]), DoubleType())
                            pred_df = pred_df.withColumn(
                                "prob_positive", extract_prob(F.col("probability"))
                            )

                            binary_eval = BinaryClassificationEvaluator(
                                labelCol=target_col,
                                rawPredictionCol="prob_positive",
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
                else:
                    auROC = float("nan")
                    auPR = float("nan")

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

                hp_performance_dict = {
                    "layers": full_layers,
                    "maxIter": hp_dict["maxIter"],
                    "stepSize": hp_dict["stepSize"],
                    "tol": hp_dict["tol"],
                    "blockSize": hp_dict["blockSize"],
                    "solver": hp_dict["solver"],
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
                    "family": None,
                    "regParam": float("nan"),
                    "fitIntercept": None,
                    "variancePower": float("nan"),
                    "linkPower": float("nan"),
                }

                val_performance.append(hp_performance_dict)
            else:
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

                # Get variancePower and linkPower values (default to NaN if not present)
                variancePower = hp_dict.get("variancePower", float("nan"))
                linkPower = hp_dict.get("linkPower", float("nan"))

                hp_performance_dict = {
                    "family": hp_dict["family"],
                    "maxIter": hp_dict["maxIter"],
                    "tol": hp_dict["tol"],
                    "regParam": hp_dict["regParam"],
                    "fitIntercept": hp_dict["fitIntercept"],
                    "solver": hp_dict["solver"],
                    "variancePower": variancePower,
                    "linkPower": linkPower,
                    "fold": k,
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
                    "layers": None,
                    "stepSize": float("nan"),
                    "blockSize": None,
                }

                val_performance.append(hp_performance_dict)

    # Define output schema
    schema = StructType(
        [
            StructField("layers", ArrayType(IntegerType()), True),
            StructField("family", StringType(), True),
            StructField("maxIter", IntegerType(), True),
            StructField("stepSize", DoubleType(), True),
            StructField("tol", DoubleType(), True),
            StructField("regParam", DoubleType(), True),
            StructField("fitIntercept", BooleanType(), True),
            StructField("blockSize", IntegerType(), True),
            StructField("solver", StringType(), True),
            StructField("variancePower", DoubleType(), True),
            StructField("linkPower", DoubleType(), True),
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


def train_mlp_model(
    training_df, params, target_col, isClassifier, pipeline_stages, weightCol=None
):
    """
    Train a Multilayer Perceptron model with specified parameters.

    Creates a complete ML pipeline by combining preprocessing stages with
    either a MultilayerPerceptronClassifier (for classification) or GeneralizedLinearRegression
    (for regression). The pipeline is fitted on the raw input data, handling all feature
    transformations and model training in a single pass.

    Args:
        training_df (DataFrame): Raw training data before preprocessing. Should contain
            all feature columns specified in the pipeline_stages, the target column,
            and optionally a weight column.

        params (dict): Hyperparameters for the model. For classification, expects:
            - layers (list, optional): Full layer specification [input, hidden..., output]
            - hidden_layers (list, optional): Just hidden layers if layers not provided
            - maxIter (int): Maximum iterations for optimization
            - stepSize (float): Learning rate for gradient descent
            - tol (float): Convergence tolerance
            - blockSize (int): Block size for stacking training data
            - solver (str): Optimization solver ('l-bfgs' or 'gd')

            For regression (GLM), expects:
            - family (str): Distribution family ('gaussian', 'gamma', 'poisson', 'tweedie')
            - maxIter (int): Maximum iterations
            - tol (float): Convergence tolerance
            - regParam (float): Regularization parameter
            - fitIntercept (bool): Whether to fit an intercept term
            - solver (str): Solver algorithm ('irls' or 'auto')
            - variancePower (float, optional): For tweedie family only
            - linkPower (float, optional): Link function power

        target_col (str): Name of the target/label column in training_df

        isClassifier (bool): True for classification tasks (uses MLP),
            False for regression tasks (uses GLM as MLP doesn't support regression)

        pipeline_stages (list): List of PySpark ML pipeline stages for preprocessing
            (e.g., StringIndexer, OneHotEncoder, VectorAssembler). These stages
            must produce a column named "final_features" containing the feature vector.

        weightCol (str, optional): Name of the column containing instance weights.
            Note: MultilayerPerceptronClassifier does not support sample weights,
            so this parameter is ignored for classification tasks.

    Returns:
        PipelineModel: Fitted pipeline model containing both preprocessing stages
            and the trained model. Can be used directly for transform() operations
            on new data.
    """
    if isClassifier:
        # First apply the pipeline to get final_features
        pipeline = Pipeline(stages=pipeline_stages)
        pipeline_model = pipeline.fit(training_df)
        preprocessed_df = pipeline_model.transform(training_df)

        # Get input feature size from preprocessed data
        sample_row = preprocessed_df.select("final_features").limit(1).collect()[0][0]
        input_size = len(sample_row)

        # Determine number of classes for output layer
        max_label = preprocessed_df.select(F.max(target_col)).first()[0]
        num_classes = int(max_label) + 1

        # If layers are not fully specified, build full layers array
        if params.get("layers") is None or len(params["layers"]) < 3:
            hidden_layers = params.get("hidden_layers", [64, 32])
            full_layers = [input_size] + hidden_layers + [num_classes]
        else:
            full_layers = params["layers"]

        # Create MultilayerPerceptronClassifier
        mlp_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "layers": full_layers,
            "maxIter": params["maxIter"],
            "stepSize": params["stepSize"],
            "tol": params["tol"],
            "blockSize": params["blockSize"],
            "solver": params["solver"],
            "seed": 42,
        }

        if weightCol is not None:
            print("Warning: MultilayerPerceptronClassifier does not support weightCol.")

        mlp = MultilayerPerceptronClassifier(**mlp_params)

        # Add MLP to pipeline stages
        pipeline_stages_with_mlp = pipeline_stages + [mlp]

        # Create and fit full pipeline
        full_pipeline = Pipeline(stages=pipeline_stages_with_mlp)
        final_model = full_pipeline.fit(training_df)

        return final_model

    else:
        # Apply pipeline first to check features for regression
        pipeline = Pipeline(stages=pipeline_stages)
        pipeline_model = pipeline.fit(training_df)

        glm_params = {
            "featuresCol": "final_features",
            "labelCol": target_col,
            "predictionCol": "prediction",
            "family": params["family"],
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "regParam": params["regParam"],
            "fitIntercept": params["fitIntercept"],
            "solver": params["solver"],
        }

        if params["family"] == "tweedie" and params.get("variancePower") is not None:
            glm_params["variancePower"] = params["variancePower"]

        if params.get("linkPower") is not None:
            glm_params["linkPower"] = params["linkPower"]

        if weightCol is not None:
            glm_params["weightCol"] = weightCol

        mlp = GeneralizedLinearRegression(**glm_params)

        # GLM to pipeline stages
        pipeline_stages_with_glm = pipeline_stages + [mlp]

        # Create and fit full pipeline
        pipeline = Pipeline(stages=pipeline_stages_with_glm)
        full_pipeline = pipeline.fit(training_df)

        return full_pipeline


def evaluate_mlp_model(
    input_df,
    params,
    full_pipeline,
    target_col,
    isClassifier,
    id_col="record",
    date_col="evaluation_date",
):
    """
    Evaluate a trained MLP model on input data

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

        pred_df = pred_df.select(
            id_col,
            date_col,
            target_col,
            "prediction",
            "probability",
        )

        if isBinary:
            # Check for both classes before calculating binary metrics
            distinct_labels = pred_df.select(target_col).distinct().count()
            distinct_preds = pred_df.select("prediction").distinct().count()

            if distinct_labels >= 2 and distinct_preds >= 2:
                try:
                    # Extract probability for positive class
                    extract_prob = F.udf(lambda v: float(v[1]), DoubleType())
                    pred_df = pred_df.withColumn(
                        "prob_positive", extract_prob(F.col("probability"))
                    )

                    binary_eval = BinaryClassificationEvaluator(
                        labelCol=target_col,
                        rawPredictionCol="prob_positive",
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
        else:
            auROC = None
            auPR = None

        # Multiclass metrics (work for both binary and multiclass)
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
            "layers": params["layers"],
            "maxIter": params["maxIter"],
            "stepSize": params["stepSize"],
            "tol": params["tol"],
            "blockSize": params["blockSize"],
            "solver": params["solver"],
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
            "family": None,
            "regParam": None,
            "fitIntercept": None,
            "variancePower": None,
            "linkPower": None,
        }
        performance.append(perf_dict)

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

        # For regression, set class metrics to None
        class_metrics_df = None
        confusion_matrix_df = None

        perf_dict = {
            "family": params["family"],
            "maxIter": params["maxIter"],
            "tol": params["tol"],
            "regParam": params["regParam"],
            "fitIntercept": params["fitIntercept"],
            "solver": params["solver"],
            "variancePower": params.get("variancePower"),
            "linkPower": params.get("linkPower"),
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
            "layers": None,
            "stepSize": None,
            "blockSize": None,
        }
        performance.append(perf_dict)

    # Define schema
    schema = StructType(
        [
            StructField("layers", ArrayType(IntegerType()), True),
            StructField("family", StringType(), True),
            StructField("maxIter", IntegerType(), True),
            StructField("stepSize", DoubleType(), True),
            StructField("tol", DoubleType(), True),
            StructField("regParam", DoubleType(), True),
            StructField("fitIntercept", BooleanType(), True),
            StructField("blockSize", IntegerType(), True),
            StructField("solver", StringType(), True),
            StructField("variancePower", DoubleType(), True),
            StructField("linkPower", DoubleType(), True),
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
    return df_performance, pred_df, class_metrics_df, confusion_matrix_df
