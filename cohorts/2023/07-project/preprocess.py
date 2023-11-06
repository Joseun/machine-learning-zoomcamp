import argparse
import os
import pickle
import warnings
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task
from tqdm import tqdm

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("amazon-products-experiment")


@task(log_prints=True)
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task(log_prints=True)
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    return df


@task(log_prints=True)
def clean_dataframe(df: pd.DataFrame):
    cleaned_data = df.dropna()
    print(cleaned_data.isnull().sum())
    cleaned_data = cleaned_data.convert_dtypes()
    print(cleaned_data.info())
    print(cleaned_data.shape)
    cleaned_data = cleaned_data.drop_duplicates()
    print(cleaned_data.shape)
    return cleaned_data


@task(log_prints=True)
def get_feature_by_user(df):
    res = list()
    for i, v in tqdm(df.groupby("UserId")):
        res.append(
            (
                i,
                len(v["ProductId"]),
                (v["Score"] == 5).sum(),
                (v["Score"] == 4).sum(),
                (v["Score"] == 3).sum(),
                (v["Score"] == 2).sum(),
                (v["Score"] == 1).sum(),
                (v["Time"].dt.dayofweek == 0).sum(),
                (v["Time"].dt.dayofweek == 1).sum(),
                (v["Time"].dt.dayofweek == 2).sum(),
                (v["Time"].dt.dayofweek == 3).sum(),
                (v["Time"].dt.dayofweek == 4).sum(),
                (v["Time"].dt.dayofweek == 5).sum(),
                (v["Time"].dt.dayofweek == 6).sum(),
                (v["Time"].dt.hour > 17).sum(),
            )
        )

    res = pd.DataFrame(
        res,
        columns=[
            "UserId",
            "revised_products",
            "5_star_df_train_gave",
            "4_star_df_train_gave",
            "3_star_df_train_gave",
            "2_star_df_train_gave",
            "1_star_df_train_gave",
            "monday_review_count_user",
            "tuesday_review_count_user",
            "wednesday_review_count_user",
            "thursday_review_count_user",
            "friday_review_count_user",
            "saturday_review_count_user",
            "sunday_review_count_user",
            "evening_reviews_by_user",
        ],
    )
    return res


@task(log_prints=True)
def get_feature_by_product(df):
    res = list()
    for i, v in tqdm(df.groupby("ProductId")):
        res.append(
            (
                i,
                len(v["UserId"]),
                (v["Score"] == 5).sum(),
                (v["Score"] == 4).sum(),
                (v["Score"] == 3).sum(),
                (v["Score"] == 2).sum(),
                (v["Score"] == 1).sum(),
                (v["Time"].dt.dayofweek == 0).sum(),
                (v["Time"].dt.dayofweek == 1).sum(),
                (v["Time"].dt.dayofweek == 2).sum(),
                (v["Time"].dt.dayofweek == 3).sum(),
                (v["Time"].dt.dayofweek == 4).sum(),
                (v["Time"].dt.dayofweek == 5).sum(),
                (v["Time"].dt.dayofweek == 6).sum(),
                (v["Time"].dt.hour > 17).sum(),
            )
        )

    res = pd.DataFrame(
        res,
        columns=[
            "ProductId",
            "user_count",
            "1_star_df_train_recieved",
            "2_star_df_train_recieved",
            "3_star_df_train_recieved",
            "4_star_df_train_recieved",
            "5_star_df_train_recieved",
            "monday_review_count_item",
            "tuesday_review_count_item",
            "wednesday_review_count_item",
            "thursday_review_count_item",
            "friday_review_count_item",
            "saturday_review_count_item",
            "sunday_review_count_item",
            "evening_reviews_by_movie",
        ],
    )
    return res


@task(log_prints=True)
def get_model_input(X_u, X_m, y, tgt_users):

    merged = pd.merge(X_u, y, on=["UserId"], how="inner")
    merged = pd.merge(X_m, merged, on=["ProductId"], how="outer")
    merged = merged.query("UserId in @tgt_users")
    # print(merged.columns)

    merged.fillna(0, inplace=True)
    features_cols = list(
        merged.drop(
            columns=[
                "UserId",
                "ProductId",
                "Score",
                "Time",
                "ProfileName",
                "Summary",
                "Text",
            ]
        ).columns
    )

    query_list = merged["UserId"].value_counts()

    merged = merged.set_index(["UserId", "ProductId"])

    query_list = query_list.sort_index()

    merged.sort_index(inplace=True)

    df_x = merged[features_cols]

    df_y = merged["Score"]

    return df_x, df_y, query_list


@flow(log_prints=True)
def run_data_prep(raw_data_path: str, dest_path: str):
    # Load csv files
    df_raw = read_dataframe(raw_data_path)

    # Clean the dataframe
    df_train = clean_dataframe(df_raw)

    start = min(df_train["Time"])
    end = max(df_train["Time"])
    interval = end - start

    df_train["Score"] = df_train["Score"].apply(lambda x: int(np.ceil(x)))

    train = df_train[df_train["Time"] <= (end - interval / 3)]
    test = df_train[df_train["Time"] >= (start + interval / 3)]

    train_y = train[train["Time"] >= (start + interval / 3)]
    train_X = train[train["Time"] < (start + interval / 3)]
    test_y = test[test["Time"] >= (end - interval / 3)]
    test_X = test[test["Time"] < (end - interval / 3)]

    train_tgt_user = set(train_X["UserId"]) & set(train_y["UserId"])
    test_tgt_user = set(test_X["UserId"]) & set(test_y["UserId"])

    train_X_u = get_feature_by_user(train_X)
    test_X_u = get_feature_by_user(test_X)

    train_X_p = get_feature_by_product(train_X)
    test_X_p = get_feature_by_product(test_X)

    X_train, y_train, query_list_train = get_model_input(
        train_X_u, train_X_p, train_y, train_tgt_user
    )
    X_test, y_test, query_list_test = get_model_input(
        test_X_u, test_X_p, test_y, test_tgt_user
    )

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save datasets
    dump_pickle(
        (X_train, y_train, query_list_train, train_X_p),
        os.path.join(dest_path, "train.pkl"),
    )
    dump_pickle(
        (X_test, y_test, query_list_test, test_X_p), os.path.join(dest_path, "val.pkl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ranking Pipeline")

    parser.add_argument("--raw_data_path", required=True, help="Location of raw data")

    parser.add_argument(
        "--dest_path", required=True, help="Destination of processed data"
    )

    args = parser.parse_args()

    run_data_prep(args.raw_data_path, args.dest_path)
