import pandas as pd
import numpy as np 
import os


def check_data():
    """ Run sanity test on dataset """
    print("=" * 60)
    print("CREDIT DEFAULT RISK - DATA SANITY CHECK")
    print("=" * 60)

    #1 Check If Data Exists
    file_path = "data/application_train.csv"

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please ensure dataset is downloaded to data/ folder")
        return False
    #2 Loading Files 
    print("\n Loading Data..." )
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    #3 Memory Usage
    memory_usage = df.memory_usage(deep=True).sum()
    memory_gb = memory_usage / (1024 ** 3)
    print(f"Memory Usage: {memory_gb:.2f}")

    #4 Target Check and Analysis
    if "TARGET" in df.columns:
        count = df["TARGET"].value_counts()
        pcts = count / count.sum() * 100

        print("\n Target Distributions: ")
        print(f"    O (Paid back): {count[0]:,} ({pcts[0]:.2f}%)")
        print(f"    1 (Defaulted): {count[1]:,} ({pcts[1]:.2f}%)")
        print(f"    Imabalance Ratio: {count[0] / count[1]:.2f}:1")
    else:
        print("Warning: Target Column Not Found")

    #5 Missing Values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total Missing Values: {total_missing:,}")

    #6 Missing Value Handling
    high_missing = missing_counts[missing_counts > 0.5 * len(df)]
    if len(high_missing) > 0:
        print(f"\n Columns with >50% missing: ")
        for col in high_missing.index:
            pct = (high_missing[col] / len(df)) * 100
            print(f"    - {col}: {pct:.1f}% missing")

    #7 Column Types
    print("\n Column Types: ")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    {dtype}: {count}")

    #8 Check for Negative Values
    if 'DAYS_BIRTH' in df.columns:
        min_age = df["DAYS_BIRTH"].min()
        max_age = df["DAYS_BIRTH"].max()
        print(f" Age Range: {abs(min_age)/365:.1f} - {abs(max_age)/365:.1f}")
        if min_age > 0:
            print(" warning: DAYS_BIRTH is positive (It should be ngative)")
    
    print("\n" + "=" * 60)
    print("Data Sanity Check Complete")
    print("=" * 60)
    return True

if __name__ == "__main__":
    check_data()
