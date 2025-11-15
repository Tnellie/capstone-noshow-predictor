from transforms.api import transform, Input, Output, configure, ComputeBackend
from prismSparkML import dimensionalityReduction as dr
from prismSparkML.modelingSupport import encode_categorical_features


@configure(
    profile=[
        "EXECUTOR_MEMORY_LARGE",
        "DRIVER_MEMORY_LARGE",
        "NUM_EXECUTORS_64",
        "EXECUTOR_MEMORY_OFFHEAP_FRACTION_MODERATE",
    ],
    backend=ComputeBackend.VELOX,
)
@transform(
    feature_importance_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    reduced_df=Output("ri.foundry.main.dataset.<your-dataset-id>"),
    encoded_df=Output("ri.foundry.main.dataset.<your-dataset-id>"),
    onehot_map=Output("ri.foundry.main.dataset.<your-dataset-id>"),
    weighted_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(weighted_df, feature_importance_df, encoded_df, onehot_map, reduced_df):
    """
    Perform feature encoding, selection, and dimensionality reduction for employee no-show prediction.

    This transform prepares ML features through a three-stage pipeline:
    1. Categorical encoding - Converts categorical features to numeric representations using one-hot encoding
    2. Feature selection - Uses Random Forest feature importance to identify the top 135 most predictive features
    3. Dimensionality reduction - Creates a reduced dataset containing only selected features
    
    The transform handles weighted samples to account for class imbalance in the no-show prediction task.
    
    Inputs:
        weighted_df: Training dataset with engineered features, target variable (noShow_day1_target),
            sample weights, and identifying columns (maskedMatchId, date). Features should include
            both numeric and categorical variables.
    
    Outputs:
        encoded_df: Full feature set after categorical encoding, with all one-hot encoded features
        onehot_map: Mapping dataframe showing the relationship between original categorical values
            and their encoded column names
        feature_importance_df: Feature importance scores from Random Forest, ranking all features
            by their predictive value for the no-show target
        reduced_df: Training dataset containing only the top 135 selected features plus identifying
            columns (maskedMatchId, date, noShow_day1_target, weight)
    
    Processing Steps:
        - Identifies feature columns by excluding metadata fields (maskedMatchId, date, noShow_day1_target, weight)
        - Applies one-hot encoding to categorical features (locationId, hrStatus) with auto-detection enabled
        - Trains Random Forest to compute feature importance scores using sample weights
        - Selects top 135 features based on importance ranking
        - Creates reduced dataset for downstream model training
    """
    # Read input DF
    weighted_df = weighted_df.dataframe()

    # Get feature cols for encoding
    feature_cols = [
        col
        for col in weighted_df.columns
        if col not in ("maskedMatchId", "date", "noShow_day1_target", "weight")
    ]
    categorical_cols = ["locationId", "hrStatus"]
    auto_detect_categorical = True
    categorical_encoding = "onehot"

    # Process categorical features
    (
        df_processed,
        processed_feature_cols,
        mapping_df,
    ) = encode_categorical_features(
        weighted_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        auto_detect_categorical=auto_detect_categorical,
        handle_method=categorical_encoding,
    )

    # Get feature cols for feature selection
    feature_cols = [
        col
        for col in df_processed.columns
        if col not in ("maskedMatchId", "date", "noShow_day1_target", "weight")
    ]

    # Select top features
    selected_features, importance_dict, importance_df = dr.select_features_rf(
        df=df_processed,
        feature_cols=feature_cols,
        target_col="noShow_day1_target",
        weight_col="weight",
        task="classification",
        n_features=135,
    )

    # Reduce training data
    keep_cols = ["maskedMatchId", "date", "noShow_day1_target", "weight"]
    reduced_training_df = dr.reduce_dataframe_rf(
        df=df_processed, selected_features=selected_features, keep_cols=keep_cols
    )

    onehot_map.write_dataframe(mapping_df)
    encoded_df.write_dataframe(df_processed)
    feature_importance_df.write_dataframe(importance_df)
    reduced_df.write_dataframe(reduced_training_df)
