'''
*** This function is called in dimensionality-reduction-cap.py ***
'''
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from typing import List, Dict, Tuple
from pyspark.sql import SparkSession


def select_features_rf(
    df: DataFrame,
    feature_cols: List[str],
    target_col: str,
    weight_col: str,
    task: str = "classification",
    n_features: int = 60,
    num_trees: int = 200,
    max_depth: int = 10,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, float], DataFrame]:
    """
    Select top features using Random Forest importance with optional class weights.
    This method of feature selection may not work well with linear models (logistic regression, SVM, etc.).
    It works best for feature selection when training tree-based models.

    Args:
        df: Input DataFrame - should already have weights added if needed for classification
        feature_cols: List of feature column names to select from
        target_col: Target column name
        weight_col: Weight column name (only used for classification)
        task: Either 'classification' or 'regression'
        n_features: Number of top features to select
        num_trees: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
            - List of selected feature names (ordered by importance)
            - Dictionary of feature importances {feature_name: importance_score}
            - DataFrame with feature importance details (for inspection)
    """
    # Validate task
    if task not in ["classification", "regression"]:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )

    df_assembled = assembler.transform(df)

    # Choose appropriate RF model
    if task == "classification":
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=target_col,
            numTrees=num_trees,
            maxDepth=max_depth,
            seed=seed,
        )
        # Add weights for classification
        if weight_col and weight_col in df.columns:
            rf.setWeightCol(weight_col)
    else:  # regression
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol=target_col,
            numTrees=num_trees,
            maxDepth=max_depth,
            seed=seed,
        )

    rf_model = rf.fit(df_assembled)

    # Get importances
    importances = rf_model.featureImportances.toArray()

    # Create importance dict
    importance_dict = {
        feature_cols[i]: float(importances[i]) for i in range(len(feature_cols))
    }

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top N
    selected_features = [feat for feat, _ in sorted_features[:n_features]]

    # Calculate metric
    total_importance = sum(importances)

    # Create DF with importance details
    spark = SparkSession.builder.getOrCreate()

    # Calculate each feature's cumulative importance
    cumulative_importance = 0
    importance_data = []

    for i, (feat, imp) in enumerate(sorted_features):
        cumulative_importance += imp
        cumulative_pct = (
            cumulative_importance / total_importance if total_importance > 0 else 0
        )

        importance_data.append(
            (
                feat,
                float(imp),
                float(imp / total_importance)
                if total_importance > 0
                else 0.0,  # Percentage
                i + 1,  # Rank
                float(cumulative_pct),  # Cumulative percentage
                "✓" if feat in selected_features else "",
            )
        )

    importance_df = spark.createDataFrame(
        importance_data,
        [
            "feature",
            "importance",
            "importance_pct",
            "rank",
            "cumulative_pct",
            "selected",
        ],
    ).sort("rank")

    return selected_features, importance_dict, importance_df


def reduce_dataframe_rf(
    df: DataFrame, selected_features: List[str], keep_cols: List[str] = None
) -> DataFrame:
    """
    Reduce DataFrame to features selected as most important by Random Forest plus additional columns.
    This function should be used with select_features_rf.

    Args:
        df: Input DataFrame
        selected_features: List of selected feature names
        keep_cols: Additional columns to keep (e.g., ['ID', 'date', 'target'])

    Returns:
        Reduced DataFrame with only selected features and keep_cols
    """
    keep_cols = keep_cols or []

    # Combine and remove duplicates
    columns_to_keep = selected_features + keep_cols
    columns_to_keep = list(
        dict.fromkeys(columns_to_keep)
    )  # Remove duplicates, preserve order

    # Filter to existing columns
    existing_cols = [col for col in columns_to_keep if col in df.columns]

    return df.select(*existing_cols)
