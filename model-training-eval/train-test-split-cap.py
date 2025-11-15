from transforms.api import transform, Input, Output, configure, ComputeBackend
from prismSparkML.upstreamSupportFunctions import (
    kfold_cv_temporal_weighting_and_stratification as cv_strat,
)


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
    training_df=Output("ri.foundry.main.dataset.d077581b-af5d-4c14-8f7d-dbb3650999b8"),
    cv_test_df=Output("ri.foundry.main.dataset.ac4d604e-adef-42de-a97a-5f75480e2df5"),
    cv_df=Output("ri.foundry.main.dataset.23e3aaab-9240-4a31-a578-709cf901782d"),
    reduced_training_df=Input(
        "ri.foundry.main.dataset.3e1713d8-6a78-49d1-9259-94b538e5f2fe"
    ),
)
def compute(reduced_training_df, training_df, cv_test_df, cv_df):
    """
    Create temporally-aware train/validation/test splits with recency-based sample weighting.

    This transform splits the reduced feature dataset into three sets for model development:
    a final training set, a cross-validation set, and a holdout test set. The splits respect
    temporal ordering to prevent data leakage, and samples are weighted based on recency to
    emphasize recent patterns in the employee no-show prediction task.

    The temporal weighting scheme applies higher weights to more recent observations, reflecting
    the assumption that recent attendance patterns are more predictive of future behavior than
    older patterns.

    Inputs:
        reduced_training_df: Dataset containing selected features, target variable 
            (noShow_day1_target), identifying columns (maskedMatchId, date), and existing
            sample weights from upstream processing.

    Outputs:
        training_df: Final training dataset with temporal weights applied, containing all data
            except the most recent 30 days. Used for final model training after hyperparameter
            tuning.
        cv_test_df: Holdout test set from the most recent 30 days of data, reserved for final
            model evaluation. Not used during cross-validation or training.
        cv_df: Cross-validation dataset with 3-fold temporal splits and stratification by target
            class. Each fold respects temporal ordering and includes fold assignments for
            hyperparameter tuning.

    Processing Details:
        - Temporal split: Most recent 30 days held out as test set, remaining data used for
        training and cross-validation
        - K-fold strategy: 3-fold cross-validation with temporal ordering and target stratification
        to handle class imbalance
        - Temporal weighting: Samples weighted by recency with a three-tier scheme:
            * Last 7 days: weight multiplier of 4x
            * 8-30 days: weight multiplier of 2x 
            * 31-90 days: weight multiplier of 1x (baseline)
        - These weights are combined with existing sample weights from upstream processing
    """
    # Read input DF
    reduced_training_df = reduced_training_df.dataframe()

    # Split data and add temporal weighting
    final_training_data, cv_testing_data, cv_data = cv_strat(
        df=reduced_training_df,
        id_col="maskedMatchId",
        timestamp_col="date",
        target_col="noShow_day1_target",
        test_period_days=30,  # Days of data held out for testing
        k_folds=3,
        weighting_period_days=[7, 30, 90],
        weighting_factor=[4, 2, 1],
    )

    training_df.write_dataframe(final_training_data)
    cv_test_df.write_dataframe(cv_testing_data)
    cv_df.write_dataframe(cv_data)
