import argparse
import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("amazon-products-experiment")

mlflow.xgboost.autolog()


@task(log_prints=True)
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task(log_prints=True)
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def coverage(preds, train_X_p):
    print(preds.columns)
    test_recs = preds["productId"].nunique()
    train_movies = train_X_p["ProductId"].nunique()

    return test_recs / train_movies

# @flow()
def predict_at_k(data, model, k: int = 5):
    user_ids = list()
    product_ids = list()
    ranks = list()

    for userId, df in data.groupby("UserId"):
        pred = model.predict(df.loc[userId])
        prodId = np.array(df.reset_index()["ProductId"])
        topK_index = np.argsort(pred)[::-1][:k]
        product_ids.extend(list(prodId[topK_index]))
        user_ids.extend([userId] * len(topK_index))
        ranks.extend(list(range(1, len(topK_index) + 1)))

    results = pd.DataFrame(
        {"userId": user_ids, "productId": product_ids, "rank": ranks}
    )

    return results

@task(log_prints=True)
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        model = xgb.XGBRanker(**params)
        model = model.fit(
            X_train,
            y_train,
            group=query_list_train,
            eval_set=[(X_val, y_val)],
            eval_group=[list(query_list_val)],
        )

        # Predicting values for training and validation data, and getting prediction probabilities
        y_pred = predict_at_k(X_val, model)
        coverage_score = coverage(y_pred, train_X_p)

        print("COVERAGE:", coverage_score)

        mlflow.log_metric("coverage", coverage_score)

    return {"loss": coverage_score, "status": STATUS_OK}


@flow(log_prints=True)
def run_optimization(data_path: str, dest_path: str, num_trials: int = 1):
    global X_train, y_train, query_list_train, train_X_p
    global X_val, y_val, query_list_val, val_X_p

    X_train, y_train, query_list_train, train_X_p = load_pickle(
        os.path.join(data_path, "train.pkl")
    )
    X_val, y_val, query_list_val, val_X_p = load_pickle(
        os.path.join(data_path, "val.pkl")
    )

    search_space = {
        "lambdarank_num_pair_per_sample": 8,
        "lambdarank_pair_method": "topk",
        "eval_metric": "ndcg",
        "tree_method": "hist",
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "objective": "rank:ndcg",
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
    )
    # Querying mlflow api instead of using web UI.
    # Sorting by validation aucroc and then getting top run for best run.
    EXPERIMENT_ID = mlflow.get_experiment_by_name(
        "amazon-products-experiment"
    ).experiment_id

    runs_df = mlflow.search_runs(
        experiment_ids=EXPERIMENT_ID, order_by=["metrics.coverage DESC"]
    )
    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]

    # Loading model from best run
    best_model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/model")
    
    y_predict_model = predict_at_k(X_val, best_model)

    # Plotting the area under the curve to visualize the performance of the model

    coverage_score = coverage(y_predict_model, train_X_p)

    print("coverage_SCORE:", coverage_score)

    markdown_coverage_report = f""" Amazon Product Coverage Report
		COVERAGE_SCORE: {coverage_score}
		"""

    create_markdown_artifact(
        key="amazon-products-coverage-report", markdown=markdown_coverage_report
    )

    recommmendations = best_model.predict(pd.concat([X_train, X_val]))
    recommmendations.to_parquet("./recommendations.parquet", index=True)
    mlflow.log_artifact(
        os.path.join(args.data_path, "recommendations.parquet"),
        artifact_path="recommendations.parquet",
    )
    mlflow.xgboost.log_model(best_model, artifact_path="artifacts/model")
    mlflow.register_model(f"runs:/{best_run_id}/artifacts/model", "AmazonProducts-XGBR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument(
        "--data_path", required=True, help="Destination of processed data"
    )

    parser.add_argument(
        "--dest_path", required=True, help="Destination of recommendations"
    )

    args = parser.parse_args()

    run_optimization(args.data_path, args.dest_path)
