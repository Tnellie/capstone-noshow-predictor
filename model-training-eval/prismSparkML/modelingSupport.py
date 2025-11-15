import random
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType, StringType, IntegerType, LongType
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StringIndexer,
    OneHotEncoder,
)
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark import keyword_only
from typing import List, Dict, Tuple, Optional
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession


def add_class_weights(
    dataframe,
    label_col="label",
    prediction_col=None,
    probability_col=None,
    date_col=None,
    important_features=None,
    weight_col_name="weight",
    strategy="inverse_class_frequency",
    custom_weights=None,
    focal_loss_gamma=2.0,
    time_decay_half_life=30,
    quantile_buckets=10,
    class_multipliers=None,
):
    """
    Apply various class imbalance handling strategies.

    Parameters:
    -----------
    dataframe : pyspark.sql.DataFrame
        Input DataFrame containing the data
    label_col : str, default="label"
        Column containing the class labels
    prediction_col : str, optional
        Column containing model predictions (required for 'focal_loss')
    probability_col : str, optional
        Column containing prediction probabilities (required for 'focal_loss')
    date_col : str, optional
        Column containing timestamps (required for 'time_based')
    important_features : list, optional
        List of tuples (feature_name, threshold) for 'feature_based' strategy
    weight_col_name : str, default="weight"
        Name of the output weight column
    strategy : str, default="inverse_class_frequency"
        Weighting strategy to use. Options:
        - "inverse_class_frequency": Weight inversely proportional to class frequency
        - "balanced": Equal importance to all classes
        - "custom": Use custom weights provided in custom_weights
        - "focal_loss": Weight based on prediction errors
        - "time_based": Weight based on recency
        - "feature_based": Weight based on feature values
        - "quantile_based": Weight based on label distribution quantiles
        - "combined": Combine multiple strategies
    custom_weights : dict, optional
        Dictionary mapping class labels to custom weights
    focal_loss_gamma : float, default=2.0
        Focusing parameter for focal loss (higher values focus more on hard examples)
    time_decay_half_life : int, default=30
        Half-life for time-based decay (in days)
    quantile_buckets : int, default=10
        Number of buckets for quantile-based weighting
    class_multipliers : dict, optional
        Additional multipliers to apply to specific classes when using 'combined' strategy

    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with added weight column
    """
    df = dataframe

    if strategy == "inverse_class_frequency":
        # Count total observations and class frequencies
        total_count = df.count()
        class_counts = df.groupBy(label_col).count()

        # Calculate weights as inverse of class frequency
        weight_df = class_counts.withColumn(
            "weight", F.lit(total_count) / F.col("count")
        )

        # Join weights back to original df
        df = df.join(weight_df.select(label_col, "weight"), on=label_col, how="left")

    elif strategy == "balanced":
        # Get number of classes & total samples
        n_samples = df.count()
        n_classes = df.select(label_col).distinct().count()

        # Get class counts
        class_counts = df.groupBy(label_col).count()

        # Calculate balanced weights
        weight_df = class_counts.withColumn(
            "weight", F.lit(n_samples) / (F.lit(n_classes) * F.col("count"))
        )

        # Join weights back to original df
        df = df.join(weight_df.select(label_col, "weight"), on=label_col, how="left")

    elif strategy == "custom":
        if not custom_weights:
            raise ValueError(
                "custom_weights parameter must be provided for 'custom' strategy"
            )

        # Create UDF for assigning custom weights
        def assign_custom_weight(label):
            return custom_weights.get(label, 1.0)

        assign_weights_udf = F.udf(assign_custom_weight, DoubleType())
        df = df.withColumn("weight", assign_weights_udf(F.col(label_col)))

    elif strategy == "focal_loss":
        if not prediction_col or not probability_col:
            raise ValueError(
                "prediction_col and probability_col must be provided for 'focal_loss' strategy"
            )

        # For binary classification
        if df.select(label_col).distinct().count() <= 2:
            # Get the probability of the true class
            df = df.withColumn(
                "prob_true_class",
                F.when(
                    F.col(label_col) == F.col(prediction_col), F.col(probability_col)
                ).otherwise(1 - F.col(probability_col)),
            )

            # Apply focal loss weighting
            df = df.withColumn(
                "weight", F.pow(1 - F.col("prob_true_class"), F.lit(focal_loss_gamma))
            ).drop("prob_true_class")
        else:
            # For multiclass, assume probability_col contains array of probabilities
            # Extract probability of true class using array indexing
            df = df.withColumn(
                "weight",
                F.pow(
                    1
                    - F.element_at(
                        F.col(probability_col), F.col(label_col).cast("int") + 1
                    ),
                    F.lit(focal_loss_gamma),
                ),
            )

    elif strategy == "time_based":
        if not date_col:
            raise ValueError("date_col must be provided for 'time_based' strategy")

        # Get maximum date
        max_date = df.agg(F.max(date_col)).collect()[0][0]

        # Calculate days from latest date & apply exponential decay
        df = (
            df.withColumn("days_old", F.datediff(F.lit(max_date), F.col(date_col)))
            .withColumn(
                "weight", F.exp(-F.col("days_old") / F.lit(time_decay_half_life))
            )
            .drop("days_old")
        )

    elif strategy == "feature_based":
        if not important_features:
            raise ValueError(
                "important_features must be provided for 'feature_based' strategy"
            )

        # Default base weight = 1.0
        df = df.withColumn("weight", F.lit(1.0))

        # Adjust weights based on important features
        for feature_name, threshold in important_features:
            df = df.withColumn(
                "weight",
                F.when(
                    F.col(feature_name) > threshold, F.col("weight") * 2.0
                ).otherwise(F.col("weight")),
            )

    elif strategy == "quantile_based":
        # For numeric labels, weight by quantile position
        if (
            df.schema[label_col].dataType == FloatType()
            or df.schema[label_col].dataType == DoubleType()
        ):
            # Calculate quantiles
            quantiles = [
                float(i) / quantile_buckets for i in range(1, quantile_buckets)
            ]

            # Get quantile values
            quantile_values = df.stat.approxQuantile(label_col, quantiles, 0.01)

            # Add 'min' and 'max' to create bucket boundaries
            min_val = df.agg(F.min(label_col)).collect()[0][0]
            max_val = df.agg(F.max(label_col)).collect()[0][0]

            splits = [min_val - 0.000001] + quantile_values + [max_val + 0.000001]

            # Create buckets
            bucketizer = Bucketizer(
                splits=splits, inputCol=label_col, outputCol="bucket"
            )

            df_bucketed = bucketizer.transform(df)

            # Count observations in each bucket
            bucket_counts = df_bucketed.groupBy("bucket").count()

            # Join bucket counts back to df and calculate inverse frequency weights
            df = (
                df_bucketed.join(bucket_counts, on="bucket", how="left")
                .withColumn(
                    "weight",
                    F.lit(df.count()) / (F.lit(quantile_buckets) * F.col("count")),
                )
                .drop("bucket", "count")
            )
        else:
            # For categorical labels, fall back to inverse class frequency
            print(
                "Warning: Label column is not numeric. Falling back to inverse class frequency."
            )
            return add_class_weights(
                dataframe=dataframe,
                label_col=label_col,
                weight_col_name=weight_col_name,
                strategy="inverse_class_frequency",
            )

    elif strategy == "combined":
        # Get class weights first (inverse frequency)
        total_count = df.count()
        class_counts = df.groupBy(label_col).count()

        weight_df = class_counts.withColumn(
            "class_weight", F.lit(total_count) / F.col("count")
        )

        # Join class weights
        df = df.join(
            weight_df.select(label_col, "class_weight"), on=label_col, how="left"
        )

        # Apply time-based weights if date_col is provided
        if date_col:
            max_date = df.agg(F.max(date_col)).collect()[0][0]
            df = df.withColumn(
                "time_weight",
                F.exp(
                    -F.datediff(F.lit(max_date), F.col(date_col))
                    / F.lit(time_decay_half_life)
                ),
            )
        else:
            df = df.withColumn("time_weight", F.lit(1.0))

        # Apply feature-based weights if important_features is provided
        if important_features:
            df = df.withColumn("feature_weight", F.lit(1.0))
            for feature_name, threshold in important_features:
                df = df.withColumn(
                    "feature_weight",
                    F.when(
                        F.col(feature_name) > threshold, F.col("feature_weight") * 1.5
                    ).otherwise(F.col("feature_weight")),
                )
        else:
            df = df.withColumn("feature_weight", F.lit(1.0))

        # Apply additional class multipliers if provided
        if class_multipliers:
            df = df.withColumn(
                "additional_multiplier",
                F.create_map(*[F.lit(x) for x in sum(class_multipliers.items(), ())]),
            ).withColumn(
                "additional_multiplier",
                F.coalesce(
                    F.col("additional_multiplier")[F.col(label_col).cast("string")],
                    F.lit(1.0),
                ),
            )
        else:
            df = df.withColumn("additional_multiplier", F.lit(1.0))

        # Combine all weights
        df = df.withColumn(
            "weight",
            F.col("class_weight")
            * F.col("time_weight")
            * F.col("feature_weight")
            * F.col("additional_multiplier"),
        ).drop("class_weight", "time_weight", "feature_weight", "additional_multiplier")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Rename weight col if desired
    if weight_col_name != "weight":
        df = df.withColumnRenamed("weight", weight_col_name)

    return df


def find_optimal_hyperparameters(cv_results_df, isClassifier, optimization_metric=None):
    """
    Find the optimal hyperparameter combination from cross-validation results.

    Parameters:
    cv_results_df: DataFrame returned from cross-validation function
    isClassifier: boolean, True for classification, False for regression
    optimization_metric: string, metric to optimize on. If None, uses the default.
                         For classification: 'auROC', 'auPR', 'accuracy', 'precision', 'recall', 'f1'
                         For regression: 'rmse', 'mae', 'r2'

    Returns:
    Dictionary with optimal hyperparameters and their performance metrics
    """
    # Set default optimization metric if None
    if optimization_metric is None:
        optimization_metric = "f1" if isClassifier else "rmse"

    # Get CV result col names
    input_col_names = cv_results_df.columns

    # Define metrics by model type
    if isClassifier:
        metric_priorities = ["auPR", "auROC", "f1", "accuracy", "precision", "recall"]
    else:
        metric_priorities = ["rmse", "mae", "r2"]

    all_metrics = [
        "auROC",
        "auPR",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "rmse",
        "mae",
        "r2",
    ]

    # Get Hyperparam col names
    exclude_cols = all_metrics + ["fold"]
    hyperparameter_cols = [name for name in input_col_names if name not in exclude_cols]

    # Calculate mean and std for each metric across folds
    agg_expressions = []
    for metric in all_metrics:
        agg_expressions.extend(
            [
                F.avg(F.col(metric)).alias(f"{metric}_mean"),
                F.stddev(F.col(metric)).alias(f"{metric}_std"),
            ]
        )

    aggregated_results = cv_results_df.groupBy(*hyperparameter_cols).agg(
        *agg_expressions
    )

    # Create ordered list of metrics to try, starting with chosen metric
    metrics_to_try = [optimization_metric]

    # Add other metrics from the priority list
    metrics_to_try.extend([m for m in metric_priorities if m != optimization_metric])

    # Initialize these variables
    optimal_row = None
    used_metric = None
    optimization_direction = None

    # Try each metric until one is not NaN
    for metric in metrics_to_try:
        minimize_metrics = ["rmse", "mae"]
        filtered_results = aggregated_results.filter(~F.isnan(F.col(f"{metric}_mean")))

        # Check for valid results after filtering
        if filtered_results.count() > 0:
            if metric in minimize_metrics:
                # Lower is better
                optimal_row = filtered_results.orderBy(
                    F.col(f"{metric}_mean").asc()
                ).first()
                optimization_direction = "minimize"
            else:
                # Higher is better
                optimal_row = filtered_results.orderBy(
                    F.col(f"{metric}_mean").desc()
                ).first()
                optimization_direction = "maximize"

            used_metric = metric
            break

    # Raise error if all metrics have NaN values
    if optimal_row is None:
        raise ValueError(
            "All metrics contain NaN values. Unable to find optimal hyperparameters."
        )

    # Get optimal hyperparam values
    optimal_params = {}
    for param in hyperparameter_cols:
        value = optimal_row[param]
        # Convert strings back to Booleans
        if param in ["fitIntercept", "standardization"]:
            if isinstance(value, str):
                optimal_params[param] = value == "True"
            else:
                optimal_params[param] = bool(value)
        else:
            optimal_params[param] = value

    # Get performance metrics
    performance_metrics = {}
    for metric in all_metrics:
        performance_metrics[f"{metric}_mean"] = optimal_row[f"{metric}_mean"]
        performance_metrics[f"{metric}_std"] = optimal_row[f"{metric}_std"]

    return {
        "optimal_hyperparameters": optimal_params,
        "performance_metrics": performance_metrics,
        "optimization_metric": used_metric,  # Return the metric actually used
        "original_metric": optimization_metric,  # Include metric requested originally
        "optimization_direction": optimization_direction,
        "optimal_score": optimal_row[f"{used_metric}_mean"],
    }


class FeatureMapStage(Transformer):
    """
    A passthrough transformer that stores feature mapping information.
    This stage doesn't modify the data but carries metadata about feature positions.
    """

    @keyword_only
    def __init__(self, feature_map=None):
        super(FeatureMapStage, self).__init__()
        self.feature_map = feature_map if feature_map is not None else {}

    def getFeatureMap(self):
        return self.feature_map

    def _transform(self, dataset):
        # Passthrough stage that doesn't modify the data
        return dataset


def create_feature_pipeline(
    continuous_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    output_col: str = "final_features",
) -> List:
    """
    Creates a PySpark ML pipeline for feature preprocessing & vectorization.
    Includes feature mapping for importance analysis.

    Args:
        continuous_cols: List of continuous feature column names (will be scaled)
        binary_cols: List of binary feature column names (0/1, no scaling)
        categorical_cols: List of categorical column names (will be one-hot encoded)
        output_col: Name of the final feature vector column (default: "final_features")

    Returns:
        List of pipeline stages ready to use with Pipeline() or pass directly to model training
    """
    pipeline_stages = []
    final_feature_cols = []

    # Feature map to identify features in the final vector
    # {vector_index: original_feature_name}
    feature_map = {}
    vector_index = 0

    # Encode categorical features
    if categorical_cols:
        for col in categorical_cols:
            indexer = StringIndexer(
                inputCol=col,
                outputCol=col + "_idx",
                handleInvalid="keep",  # Handle unknown categories
            )
            encoder = OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_vec")
            pipeline_stages += [indexer, encoder]
            final_feature_cols.append(col + "_vec")
            feature_map[vector_index] = f"{col}_encoded"
            vector_index += 1

    # Scale continuous features
    if continuous_cols:
        cont_assembler = VectorAssembler(
            inputCols=continuous_cols,
            outputCol="assembled_continuous",
            handleInvalid="skip",  # Skip rows with invalid values
        )
        scaler = StandardScaler(
            inputCol="assembled_continuous",
            outputCol="scaled_continuous",
            withStd=True,
            withMean=True,
        )
        pipeline_stages += [cont_assembler, scaler]
        final_feature_cols.append("scaled_continuous")

        # Map continuous features to their positions in the final vector
        for col in continuous_cols:
            feature_map[vector_index] = col
            vector_index += 1

    # Handle binary features - no scaling
    if binary_cols:
        binary_assembler = VectorAssembler(
            inputCols=binary_cols, outputCol="assembled_binary", handleInvalid="skip"
        )
        pipeline_stages.append(binary_assembler)
        final_feature_cols.append("assembled_binary")

        # Map binary features to their positions in the final vector
        for col in binary_cols:
            feature_map[vector_index] = col
            vector_index += 1

    # Assemble features
    if final_feature_cols:
        final_assembler = VectorAssembler(
            inputCols=final_feature_cols, outputCol=output_col
        )
        pipeline_stages.append(final_assembler)
    else:
        raise ValueError(
            "At least one feature type (continuous, binary, or categorical) must be provided"
        )

    # Add custom stage to store the feature map
    feature_map_stage = FeatureMapStage(feature_map=feature_map)
    pipeline_stages.append(feature_map_stage)

    return pipeline_stages


def _categorical_mapping_to_dataframe(
    categorical_mapping: Dict[str, List[str]],
    category_labels: Dict[str, List[str]],
    spark: SparkSession = None,
) -> DataFrame:
    """
    Convert categorical feature mapping dictionary to PySpark DataFrame.

    Args:
        categorical_mapping: Dictionary from preprocess_categorical_features
                           {original_col: [expanded_col1, expanded_col2, ...]}
        category_labels: Dictionary mapping original column to list of category values
                        {original_col: [category_value1, category_value2, ...]}
        spark: SparkSession (optional, will get or create if not provided)

    Returns:
        DataFrame with cols: original_feature, encoded_feature, encoding_index,
                           category_value, total_encoded_features

    Used by encode_categorical_features
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    # Flatten the dict into rows
    mapping_data = []

    for original_feature, encoded_features in categorical_mapping.items():
        category_values = category_labels.get(original_feature, [])

        for idx, encoded_feature in enumerate(encoded_features):
            # Get the actual category value for index
            category_value = (
                category_values[idx] if idx < len(category_values) else None
            )

            mapping_data.append(
                (
                    original_feature,
                    encoded_feature,
                    idx,
                    category_value,
                    len(encoded_features),  # Total num of encoded features
                )
            )

    mapping_df = spark.createDataFrame(
        mapping_data,
        [
            "original_feature",
            "encoded_feature",
            "encoding_index",
            "category_value",
            "total_encoded_features",
        ],
    )

    return mapping_df


def encode_categorical_features(
    df: DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str] = None,
    auto_detect_categorical: bool = True,
    max_categories: int = 50,
    handle_method: str = "onehot",  # 'onehot' or 'label'
) -> Tuple[DataFrame, List[str], DataFrame]:
    """
    Encodes categorical variables before feature selection.

    Automatically detects and encodes categorical features, expanding them
    into numeric representations suitable for VectorAssembler.

    Args:
        df: Input DataFrame
        feature_cols: List of all feature column names
        categorical_cols: Explicit list of categorical columns (optional)
        auto_detect_categorical: If True, automatically detect categorical columns
        max_categories: Maximum unique values to consider a column categorical
        handle_method: How to encode categoricals:
            - 'onehot': One-hot encoding (best for most cases)
            - 'label': Label encoding (only for tree models)

    Returns:
        Tuple of:
            - Preprocessed DataFrame with encoded features
            - Updated list of feature column names (with expanded categoricals)
            - Mapping DataFrame with one hot feature column mapping including category values
    """
    # Identify categorical cols
    if categorical_cols is None and auto_detect_categorical:
        categorical_cols = []

        for col_name in feature_cols:
            col_type = df.schema[col_name].dataType

            # Check for string type
            if isinstance(col_type, StringType):
                categorical_cols.append(col_name)

            # Check if integer/long with limited unique values
            elif isinstance(col_type, (IntegerType, LongType)):
                n_unique = df.select(col_name).distinct().count()
                if n_unique <= max_categories:
                    categorical_cols.append(col_name)

    elif categorical_cols is None:
        categorical_cols = []

    #
    if not categorical_cols:
        return df, feature_cols, None

    # Encode categorical features
    encoded_df = df
    new_feature_cols = [col for col in feature_cols if col not in categorical_cols]
    categorical_mapping = {}
    category_labels = {}

    if handle_method == "onehot":
        for cat_col in categorical_cols:
            # String indexing
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_indexed",
                handleInvalid="keep",  # Keep unknown values
            )
            indexer_model = indexer.fit(encoded_df)
            encoded_df = indexer_model.transform(encoded_df)

            # Extract category values from fitted model
            labels = indexer_model.labels
            category_labels[cat_col] = labels

            # One-hot encoding
            encoder = OneHotEncoder(
                inputCol=f"{cat_col}_indexed",
                outputCol=f"{cat_col}_vec",
                dropLast=False,  # Keep all categories for feature selection
            )
            encoded_df = encoder.fit(encoded_df).transform(encoded_df)

            # Convert vector to array
            encoded_df = encoded_df.withColumn(
                f"{cat_col}_array", vector_to_array(f"{cat_col}_vec")
            )

            # Get number of categories
            n_categories = len(labels)

            # Extract individual binary cols
            expanded_cols = []
            for i in range(n_categories):
                new_col_name = f"{cat_col}_cat{i}"
                encoded_df = encoded_df.withColumn(
                    new_col_name, F.col(f"{cat_col}_array")[i].cast("double")
                )
                expanded_cols.append(new_col_name)
                new_feature_cols.append(new_col_name)

            categorical_mapping[cat_col] = expanded_cols

            # Drop intermediate cols
            encoded_df = encoded_df.drop(
                f"{cat_col}_indexed", f"{cat_col}_vec", f"{cat_col}_array", cat_col
            )
    elif handle_method == "label":
        for cat_col in categorical_cols:
            indexer = StringIndexer(
                inputCol=cat_col, outputCol=f"{cat_col}_label", handleInvalid="keep"
            )
            indexer_model = indexer.fit(encoded_df)
            encoded_df = indexer_model.transform(encoded_df)

            # Extract labels for label encoding
            labels = indexer_model.labels
            category_labels[cat_col] = labels

            new_feature_cols.append(f"{cat_col}_label")
            categorical_mapping[cat_col] = [f"{cat_col}_label"]

            # Drop original categorical cols
            encoded_df = encoded_df.drop(cat_col)
    else:
        raise ValueError(
            f"Unknown handle_method: '{handle_method}'. Use 'onehot' or 'label'"
        )

    # Create mapping DataFrame with category values
    mapping_df = _categorical_mapping_to_dataframe(
        categorical_mapping, category_labels, encoded_df.sparkSession
    )

    return encoded_df, new_feature_cols, mapping_df


def random_parameter_search(param_space, num_samples=50, seed=42):
    """
    Random parameter search function that works with any algorithm.

    Args:
        param_space: Dictionary where keys are parameter names and values are lists of parameter values
        num_samples: Number of hyperparameter combinations to generate
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries, each containing a parameter combination
    """
    random.seed(seed)

    # Get parameter names
    param_names = list(param_space.keys())

    # Tracking unique combinations using a set for O(1) lookup
    unique_combinations = set()

    # Set a max attempt limit to avoid infinite loops
    max_attempts = num_samples * 10
    attempts = 0

    while len(unique_combinations) < num_samples and attempts < max_attempts:
        # Generate parameter values
        values = []
        for name in param_names:
            value = random.choice(param_space[name])
            # Convert lists to tuples for hashability
            if isinstance(value, list):
                value = tuple(value)
            values.append(value)

        # Create hashable combo representation
        combo_tuple = tuple(values)

        # Add to set if it's unique
        unique_combinations.add(combo_tuple)

        attempts += 1

    # Convert tuples to dicts
    result = []
    for combo in unique_combinations:
        combo_dict = {}
        for i, name in enumerate(param_names):
            value = combo[i]
            # Convert tuples back to lists if needed
            if isinstance(value, tuple):
                value = list(value)
            combo_dict[name] = value
        result.append(combo_dict)

    # Warn if unable to generate enough combos
    if len(result) < num_samples:
        print(
            f"Warning: Could only generate {len(result)} unique combinations after {attempts} attempts."
        )

    return result
