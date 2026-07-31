import pandas as pd 
import argparse
import os

def load_and_merge(force_rebuild = False):
    """ Loading Data from application_train.csv, bureau.csv, previous_application.csv """
    cached_path = "data/processed/merged_data.parquet"
    if os.path.exists(cached_path) and not force_rebuild:
        print("\nLoading previosuly saved merged_data.parquet")
        df = pd.read_parquet(cached_path)
        return df
    print("=" * 60)
    print("Loading and Extracting Data from files")
    print("=" * 60)

    #1.0 Load Data from application_train.csv
    print("\n Loading data from application_train.csv")
    app = pd.read_csv('data/application_train.csv')
    print(f"Shape: {app.shape[0]:,} rows, {app.shape[1]:,} columns")

    #1.1 Load Data from bureau.csv
    print("\n Loading Data from bureau.csv")
    bureau = pd.read_csv('data/bureau.csv')
    print(f"Shape: {bureau.shape[0]:,} rows, {bureau.shape[1]:,} columns")

    #1.2 Loading data from previous_application.csv
    print("\n Loading Data from previous_application.csv")
    prev = pd.read_csv('data/previous_application.csv')
    print(f"Shape: {prev.shape[0]:,} rows, {prev.shape[1]:,} columns")

    #2 Aggregating Data in the bureau.csv file to match granularity of client level for application.csv
    print("\n Aggregating bureau.csv.....")
    bureau_agg = bureau.groupby("SK_ID_CURR").agg({"AMT_CREDIT_SUM": ["sum","mean","max"], 
    "CNT_CREDIT_PROLONG": ["sum"], "DAYS_CREDIT": ["min","mean"], "AMT_CREDIT_SUM_DEBT": ["sum","mean"],
    "AMT_CREDIT_SUM_OVERDUE": ["sum"]}).reset_index()

    #Renaming the columns
    bureau_agg.columns = ["SK_ID_CURR", "BUREAU_CREDIT_SUM", "BUREAU_CREDIT_MEAN", "BUREAU_CREDIT_MAX", 
    "BUREAU_PROLONG_SUM", "BUREAU_DAYS_MIN", "BUREAU_DAYS_MEAN", "BUREAU_DEBT_SUM",
    "BUREAU_DEBT_MEAN", "BUREAU_OVERDUE_SUM"]
    print(f"Aggregated to {bureau_agg.shape[0]:,} clients")

    #3 Aggregating Data in the previous_applications.csv
    print("\n Aggregating prev_applications.csv....")
    prev_agg = prev.groupby("SK_ID_CURR").agg({"AMT_ANNUITY": ["sum","mean"], "AMT_CREDIT": ["sum","mean"],
    "AMT_APPLICATION": ["sum","mean"], "AMT_GOODS_PRICE": ["sum","mean"], "DAYS_DECISION": ["min","mean"],
    "CNT_PAYMENT": ["sum","mean"]}).reset_index()

    prev_agg.columns = ["SK_ID_CURR", "PREV_ANNUITY_SUM", "PREV_ANNUITY_MEAN", "PREV_CREDIT_SUM",
    "PREV_CREDIT_MEAN", "PREV_APPLICATION_SUM", "PREV_APPLICATION_MEAN", "PREV_GOODS_SUM", "PREV_GOODS_MEAN",
    "PREV_DECISION_MIN", "PREV_DECISION_MEAN", "PREV_PAYMENT_SUM", "PREV_PAYMENT_MEAN"]
    print(f"Aggregated to {prev_agg.shape[0]:,} clients")

    #4 Merging Application.csv with bureau first than with prev_applications.csv
    print("\n Merging Aggregated previous_applications.csv and bureau.csv data with main application.csv data")
    df = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")

    print(f"Final Shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

    #5 Save Merged Dataframe for faster loading
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/merged_data.parquet", index=False)
    print("\n Save merged data to data/processed/merged_data.parquet")

    return df 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and Merge Data")
    parser.add_argument("--rebuild",action="store_true", help = "Force rebuild the parquet file from scratch")
    args = parser.parse_args()

    df = load_and_merge(force_rebuild = args.rebuild)
    print("\n Merge Complete")