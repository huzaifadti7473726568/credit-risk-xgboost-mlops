import os
import joblib
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from transformers import (
    MissingIndicator,
    KFoldTargetEncoder,
    WeightOfEvidenceEncoder,
    OutlierCapper,
    FrequencyEncoder,
    DropColumns
)


def build_preprocessing_pipeline(num_cols, cat_cols, use_freq=True, use_woe=True):
    """Construct memory optimized scikit-learn pipeline """

    flag_cols = [f'{col}_is_missing' for col in num_cols]

    Numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median').set_output(transform='pandas')),
        ('outlier_capper',OutlierCapper(columns=num_cols)),
        ('scaler', StandardScaler().set_output(transform='pandas'))
    ])

    cat_steps = [('imputer', SimpleImputer(strategy='constant', fill_value='MISSING').set_output(transform='pandas'))]

    if use_freq:
        cat_steps.append(('freq_encoder', FrequencyEncoder(columns=cat_cols)))
    if use_woe:
        cat_steps.append(('woe_encoder', WeightOfEvidenceEncoder(columns=cat_cols, smoothing=0.5, n_folds=5)))

    cat_steps.append(('target_encoder', KFoldTargetEncoder(columns=cat_cols, smoothing=10, n_folds=5)))
    cat_steps.append(('drop_raw_categoricals', DropColumns(columns=cat_cols)))

    categorical_transformer = Pipeline(steps=cat_steps)

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', Numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
            ('flags', 'passthrough', flag_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    master_pipeline = Pipeline(steps=[
        ('global_missing_flag', MissingIndicator(columns=num_cols)),
        ('router', col_transformer)
    ])

    return master_pipeline

def run_ablation_study():
    """ 
    Reads the data exactly once, tests multiple preprocessing configurations
    and safely saves the numpy arrays to disk for model training
    """
    print("\n" + "="*50)
    print("INITIATING DATA PREPROCESSING PIPELINE")
    print("="*50)
#___1 Loading Data Once....Memory Optimization
    print("\nLoading parquet files into memory...")
    X_train = pd.read_parquet('data/processed/split/X_train.parquet')
    y_train = pd.read_parquet('data/processed/split/y_train.parquet').squeeze()

    X_val = pd.read_parquet('data/processed/split/X_val.parquet')
    y_val = pd.read_parquet('data/processed/split/y_val.parquet').squeeze()

    X_test = pd.read_parquet('data/processed/split/X_test.parquet')
    y_test = pd.read_parquet('data/processed/split/y_test.parquet').squeeze()

    #2____Defining the Schema
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if "SK_ID_CURR" in num_cols:
        num_cols.remove("SK_ID_CURR")
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    print(f"Schema: {len(num_cols)} Numeric Features || {len(cat_cols)} Categorical Features")

    #___3 The Ablation Grid
    config = [
        ('target_only', False, False),
        ('freq_target', True, False),
        ('woe_target', False, True),
        ('all_three', True, True)
    ]

    for tags, use_freq, use_woe in config:
        print(f"Run {tags}: Frequency: {use_freq}  | Woe: {use_woe}")

        pipeline = build_preprocessing_pipeline(
            num_cols=num_cols,
            cat_cols=cat_cols,
            use_freq=use_freq,
            use_woe=use_woe,
        )

        #___4 Fit and Trasnform
        print(f"Fitting Pipeline on training data (Implementing Stratified KFold)")
        X_train_processed = pipeline.fit_transform(X_train, y_train)

        print(f"Transforing validation and test data")
        X_val_processed = pipeline.transform(X_val)
        X_test_processed =  pipeline.transform(X_test)

        #Cast to float32 to drastically save Ram
        X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
        X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
        X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

        print(f"Matrix Shapes: Train: {X_train_processed.shape} | Val: {X_val_processed.shape}")

        #___5 Save Artifacts safely
        os.makedirs('models', exist_ok=True)
        out_dir = f"data/processed/processed_splits_{tags}"
        os.makedirs(out_dir, exist_ok=True)

        # Save the machine
        joblib.dump(pipeline, f'models/preprocessor_{tags}.joblib')

        #Save the feature names for SHAP explainabaility later
        feature_names = pipeline.get_feature_names_out()
        with open(f'models/feature_names_{tags}.txt','w') as f:
            f.write('\n'.join(feature_names))


        #Save the arrays
        np.save(f"{out_dir}/X_train.npy", X_train_processed)
        np.save(f"{out_dir}/X_val.npy", X_val_processed)
        np.save(f"{out_dir}/X_test.npy", X_test_processed)

        np.save(f"{out_dir}/y_train.npy", y_train.values)
        np.save(f"{out_dir}/y_val.npy", y_val.values)
        np.save(f"{out_dir}/y_test.npy", y_test.values)

        print(f" [Sucess] Processed Arrays Saved in {out_dir}")
    
    print("\n" + "="*50)
    print("ALL ABLATION CONFIGURATIONS COMPLETED")
    print("="*50)

if __name__ == "__main__":
    run_ablation_study()
















