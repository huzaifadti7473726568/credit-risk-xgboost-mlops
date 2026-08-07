import pandas as pd
from sklearn.model_selection import train_test_split 
import os

def split_data():
    """ Load Merged Data and split to train/val/test"""
    print("=" * 60)
    print("Spliting Data to train/val/test sets")
    print("=" * 60)

    #1 Load Data from path
    print("Loading merged data")
    data_path = "data/processed/merged_data.parquet"
    df = pd.read_parquet(data_path)
    print(f" Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    #2 Seperating Targets and Features
    print("\n Seperating Targets and Features")
    X = df.drop(columns=["SK_ID_CURR","TARGET"])
    y = df["TARGET"]
    print(f"Number of Features: {X.shape[1]}")
    print(f"Target: {y.shape[0]:,} rows")

    #3 Splitting data to train and test sets
    print("\n Splitting dataset(stratified) to train, validation and test")
    X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.4, random_state=42, stratify=y)
    # splitting temp(val+test) dataset to val and test datasets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    #4 Verifying stratification Rates across all splits(train, val, test)
    print("\n verifying stratification rates (default_averages) across all splits (target)")
    print(f"Train set   {y_train.mean():.3%}")
    print(f"Val set     {y_val.mean():.3%}")
    print(f"Test set    {y_test.mean():.3%}")
    print(f"Main set    {y.mean():.3%}")

    #5 Verifying Shapes across all
    print("\n Shape verificaton of train/val/test sets")
    print(f"Train Set: {X_train.shape[0]:,} rows, {X_train.shape[1]} columns")
    print(f"Test Set: {X_test.shape[0]:,} rows, {X_test.shape[1]} columns")
    print(f"Val Set: {X_val.shape[0]:,} rows, {X_val.shape[1]} columns")

    #6 Saving Splits to data/processed
    os.makedirs("data/processed/split", exist_ok=True)

    X_train.to_parquet("data/processed/split/X_train.parquet", index=False)
    X_test.to_parquet("data/processed/split/X_test.parquet", index=False)
    X_val.to_parquet("data/processed/split/X_val.parquet", index=False)

    y_train.to_frame().to_parquet("data/processed/split/y_train.parquet", index=False)
    y_test.to_frame().to_parquet("data/processed/split/y_test.parquet", index=False)
    y_val.to_frame().to_parquet("data/processed/split/y_val.parquet", index=False)

    print(" Splits saved to data/processed/split")
    with open("data/processed/split/feature_names.txt", "w") as f:
        f.write("\n".join(X_train.columns))
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()
    print("\n Split Complete")