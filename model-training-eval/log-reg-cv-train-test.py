from pyspark.sql import functions as F
from transforms.api import transform, Input, Output, configure, ComputeBackend
from prismSparkML.modelingSupport import (
    find_optimal_hyperparameters,
    create_feature_pipeline,
)
from prismSparkML import LogisticRegression as lr


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
    cv_results=Output("ri.foundry.main.dataset.<your-dataset-id>"),
    performance_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    predictions_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    conf_matrix_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    class_metrics_df=Output(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
    cv_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    cv_testing_df=Input("ri.foundry.main.dataset.<your-dataset-id>"),
    final_train_df=Input(
        "ri.foundry.main.dataset.<your-dataset-id>"
    ),
)
def compute(
    cv_df,
    cv_testing_df,
    final_train_df,
    cv_results,
    performance_df,
    predictions_df,
    class_metrics_df,
    conf_matrix_df,
):
    """
    Perform hyperparameter tuning and validation for Logistic Regression employee no-show prediction.
    
    This transform executes a comprehensive machine learning workflow for binary classification:
    1. Hyperparameter tuning via k-fold cross-validation with grid search
    2. Model training using optimal hyperparameters on the full training set
    3. Final model evaluation on the temporal holdout test set
    
    The transform handles extreme class imbalance (0.15% no-show rate) through weighted samples,
    classification threshold tuning, and careful evaluation using precision-recall metrics rather
    than accuracy-based metrics.
    
    Inputs:
        cv_df: Cross-validation dataset with 3-fold splits, temporal weighting, and stratification.
            Contains fold assignments (fold_id), sample weights, and all selected features.
        cv_testing_df: Combined training and test dataset with is_test flag (0=train, 1=test).
            Used for final model training and holdout evaluation.
        final_train_df: Full training dataset for final model training (currently unused in code,
            reserved for production model training after validation).
    
    Outputs:
        cv_results: Cross-validation results containing performance metrics for all hyperparameter
            combinations across all folds. Includes metrics like AUPR, AUROC, precision, recall
            for each parameter configuration.
        performance_df: Aggregate performance metrics on the holdout test set, including AUPR
            (primary optimization metric), AUROC, and threshold-dependent metrics.
        predictions_df: Row-level predictions on the holdout test set with predicted probabilities,
            binary predictions, actual labels, and identifying columns (maskedMatchId, date).
        class_metrics_df: Per-class performance metrics (precision, recall, F1) for both the
            no-show and show classes on the holdout test set.
        conf_matrix_df: Confusion matrix showing true positives, false positives, true negatives,
            and false negatives on the holdout test set.
    """
    # Define cols to ignore & binary cols
    ignore_cols = [
        "maskedMatchId",
        "date",
        "noShow_day1_target",
        "weight",
        "days_from_reference",
        "dup_factor",
        "timestamp_bin",
        "target_bin",
        "strat_group",
        "fold_id",
        "timestamp_bin_edges",
        "target_bin_edges",
        "days_from_reference",
    ]

    all_binary_cols = [
        "locationId_cat0",
        "locationId_cat1",
        "locationId_cat2",
        "hrStatus_cat0",
        "hrStatus_cat1",
        "hrStatus_cat2",
        "noShowBeforeWeekend",
        "noShowAfterWeekend",
        "isWeekend",
    ]

    ignore_all = ignore_cols + all_binary_cols

    # Read input DF
    cv_df = cv_df.dataframe()

    # Define feature types for preprocessing
    continuous_cols = [col for col in cv_df.columns if col not in ignore_all]
    binary_cols = [col for col in cv_df.columns if col in all_binary_cols]

    # Create cv preprocessing pipeline
    cv_pipeline = create_feature_pipeline(
        continuous_cols=continuous_cols, binary_cols=binary_cols
    )

    # Logisitc Regression cross-validation hyperparameter search
    cv_result_df = lr.CV_LogisticRegression(
        cv_df=cv_df,
        target_col="noShow_day1_target",
        isClassifier=True,
        pipeline_stages=cv_pipeline,
        weightCol="weight",
        fold_col="fold_id",
        id_col="maskedMatchId",
        date_col="date",
        n_param_combos=4,
        param_overrides={
            "elasticNetParam": [0.0, 0.1, 0.25, 0.5, 0.75],
            "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "regParam": [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
        },
    )

    # Write CV results to DF
    cv_results.write_dataframe(cv_result_df)

    optimal_results = find_optimal_hyperparameters(
        cv_results_df=cv_result_df, isClassifier=True, optimization_metric="auPR"
    )
    # Get optimal params
    optimal_params = optimal_results["optimal_hyperparameters"]

    # Clear memory
    cv_df.unpersist()
    del cv_df, cv_pipeline, cv_result_df

    # Read input DF
    cv_testing_df = cv_testing_df.dataframe()

    # Split training and test data
    train_data = cv_testing_df.filter(F.col("is_test") == 0)
    test_data = cv_testing_df.filter(F.col("is_test") == 1)

    # Cache test_data for use during evaluation
    test_data.cache()

    # Create validation pipeline
    validation_pipeline = create_feature_pipeline(
        continuous_cols=continuous_cols, binary_cols=binary_cols
    )

    # Train model using optimal hyperparameters
    trained_model = lr.train_lr_model(
        training_df=train_data,
        params=optimal_params,
        target_col="noShow_day1_target",
        isClassifier=True,
        pipeline_stages=validation_pipeline,
        weightCol="weight",
    )

    (
        test_performance,
        test_predictions,
        class_metrics,
        confusion_matrix,
    ) = lr.evaluate_lr_model(
        input_df=test_data,
        params=optimal_params,
        full_pipeline=trained_model,
        target_col="noShow_day1_target",
        isClassifier=True,
        id_col="maskedMatchId",
        date_col="date",
    )

    # Write test results to DFs
    performance_df.write_dataframe(test_performance)
    predictions_df.write_dataframe(test_predictions)
    class_metrics_df.write_dataframe(class_metrics)
    conf_matrix_df.write_dataframe(confusion_matrix)

    # Clear memory
    test_data.unpersist()
    cv_testing_df.unpersist()
    del cv_testing_df, train_data, test_data, trained_model
    del validation_pipeline, test_performance, test_predictions

    # # Read input DF
    # final_train_df = final_train_df.dataframe()

    # # Create final model pipeline
    # final_pipeline = create_feature_pipeline(
    #     continuous_cols=continuous_cols, binary_cols=binary_cols
    # )

    # # Train final model
    # final_lr_model = lr.train_lr_model(
    #     training_df=final_train_df,
    #     params=optimal_params,
    #     target_col="noShow_day1_target",
    #     isClassifier=True,
    #     pipeline_stages=final_pipeline,
    #     weightCol="weight",
    # )
