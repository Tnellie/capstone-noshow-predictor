import pyspark.sql.functions as F
from transforms.api import transform_df, Input, Output, configure, ComputeBackend
from prismSparkML.modelingSupport import add_class_weights


@configure(
    profile=[
        "EXECUTOR_MEMORY_LARGE",
        "DRIVER_MEMORY_LARGE",
        "NUM_EXECUTORS_64",
        "EXECUTOR_MEMORY_OFFHEAP_FRACTION_MODERATE",
    ],
    backend=ComputeBackend.VELOX,
)
@transform_df(
    Output("ri.foundry.main.dataset.<your-dataset-id>"),
    training_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
)
def compute(training_df):
    """
    Add class weights to training data for handling imbalanced no-show predictions.

    Applies a selected weighting methodology to address class imbalance
    in the employee no-show dataset. The weighted data is then sorted chronologically
    to preserve time series ordering, which is critical for temporal feature engineering
    and model training.

    Process:
    1. Apply inverse class frequency weighting to balance no-show vs show classes
    2. Sort data by employee (maskedMatchId) and date to maintain temporal order

    Args:
        training_df: Training dataset containing employee attendance records with
                    features and no-show target variable

    Returns:
        DF with added 'weight' column and 'ID' column, sorted chronologically
        by employee and date. Weight values are higher for minority class (no-shows)
        to balance class representation during model training.

    Output Columns:
        - All original columns from training_df
        - weight (float): Class weight for balancing (higher for no-shows)
    """
    training_df = training_df.drop(
        "noShow_day2_target",
        "noShow_day3_target",
        "noShow_day4_target",
        "noShow_day5_target",
        "noShow_day6_target",
        "noShow_day7_target",
    )

    result_df = add_class_weights(
        training_df,
        label_col="noShow_day1_target",
        strategy="inverse_class_frequency",
    )

    # Reorder rows to maintain temporal integrity
    result_df = result_df.orderBy(F.col("maskedMatchId").asc(), F.col("date").asc())

    return result_df
