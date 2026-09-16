import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyEncoder(BaseEstimator,TransformerMixin):
    """ Replaces categories with their frequency in the training set """
    def __init__(self, columns = None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.freq_maps_ = {}
        for col in self.columns:
            self.freq_maps_[col] = X[col].value_counts().to_dict()
        return self
    
    def transform(self, X):
        check_is_fitted(self, "freq_maps_")
        X = X.copy()
        freq_cols = {
            f"{col}_freq" : X[col].map(self.freq_maps_[col]).fillna(0)
            for col in self.columns
        }
        df = pd.DataFrame(freq_cols,index=X.index) 
        X = pd.concat([X,df], axis=1)       
        return X
    
    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.columns
        return np.array(base +[f"{col}_freq" for col in self.columns])

class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoding for categorical columns"""
    """self.smoothing here is Laplace/additive smoothing on the good/bad counts -- NOT the same kind
    of quantity as KfoldTargetEncoder.smoothing below (that one is a pseudo-observation-count in a shrinkage-toward-the-mean formula)"""
    def __init__(self, columns=None, smoothing=0.5, n_folds=5):
        self.columns = columns
        self.smoothing = smoothing
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.woe_maps_ = {}
        total_bad = y.sum()
        total_good = len(y) - total_bad

        for col in self.columns:
            temp = X[[col]].copy()
            temp['target'] = y.values
            grouped = temp.groupby(col)['target'].agg(['sum', 'count'])
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']

            #Smoothing and finding log odds of distribution of good and bad for each category
            num_categories = len(grouped)
            grouped['bad_ratio'] = (grouped['bad'] + self.smoothing) / (total_bad + self.smoothing*num_categories)
            grouped['good_ratio'] = (grouped['good'] + self.smoothing) / (total_good + self.smoothing*num_categories)

            #Calculating WOE from Distribution of Goods and Bads
            grouped['woe'] = np.log(grouped['good_ratio'] / grouped['bad_ratio'])

            self.woe_maps_[col] = grouped['woe'].to_dict()

        return self
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        X = X.copy()

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        encoded_col = {}

        for col in self.columns:
            col_oof = np.zeros(X.shape[0])

            for train_idx, val_idx in kf.split(X, y):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]

                fold_total_bad = y_train_fold.sum()
                fold_total_good = len(y_train_fold) - fold_total_bad

                temp = X_train_fold[[col]].copy()
                temp['target'] = y_train_fold.values
                fold_grouped = temp.groupby(col)['target'].agg(['sum', 'count'])
                fold_grouped.columns = ['bad', 'total']
                fold_grouped['good'] = fold_grouped['total'] - fold_grouped['bad']

                num_categories = len(fold_grouped)
                fold_grouped['bad_ratio'] = (fold_grouped['bad'] + self.smoothing) / (fold_total_bad + num_categories*self.smoothing)
                fold_grouped['good_ratio'] = (fold_grouped['good'] + self.smoothing) / (fold_total_good + num_categories*self.smoothing)
                fold_grouped['woe'] = np.log(fold_grouped['good_ratio']/fold_grouped['bad_ratio'])

                fold_encoding = fold_grouped['woe'].to_dict()
                col_oof[val_idx] = X.iloc[val_idx][col].map(fold_encoding).fillna(0)

            encoded_col[f"{col}_woe"] = col_oof
        df = pd.DataFrame(encoded_col, index=X.index)
        X = pd.concat([X,df], axis=1)

        return X

    def transform(self, X):
        check_is_fitted(self, "woe_maps_")
        X = X.copy()
        woe_col = {
            f"{col}_woe" : X[col].map(self.woe_maps_[col]).fillna(0)
            for col in self.columns
        }
        df = pd.DataFrame(woe_col, index=X.index)
        X = pd.concat([X,df], axis=1)
        return X
    
    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.columns
        return np.array(base + [f"{col}_woe" for col in self.columns])

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Target ENcoder with K-Fold cross validation to prevent data leakage.
    fit_transform(X,y): Use on training data only.
    - Compute leak-free out-of-fold encoding for training row (each row is encoded using 
    stats from every other fold.) 
    - Also run fit(X,y) as side effect which stores full_data encoding dict for later 
    use by transform() function

    transform(X): use on new/unseen data (validation/test/production)
    - No y required. Look up each row's category in the encoding_dict that was saved
    when fit(X,y) ran. COmpletely safe as these rows dont leak data to dict.
    """

    def __init__(self, columns=None, n_folds=5, smoothing=10):
        self.columns = columns
        self.n_folds = n_folds
        self.smoothing = smoothing

    def fit(self, X, y):
        self.encoded_map_ = {}
        self.global_mean_ = y.mean()
        for col in self.columns:
            temp = X[[col]].copy()
            temp['target'] = y.values
            stat = temp.groupby(col)['target'].agg(['mean','count'])
            #Calculating weighted average of categories in column + global mean (for smoothing)
            stat['encoding'] = (
                (stat['count'] * stat['mean'] + self.smoothing * self.global_mean_) / 
                (stat['count'] + self.smoothing)
            )
            self.encoded_map_[col] = stat['encoding'].to_dict()      
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        X = X.copy()

        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,random_state=42)
        encoded_col = {}

        for col in self.columns:
            col_oof = np.zeros(X.shape[0])

            for train_idx, val_idx in kf.split(X, y):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                fold_global_mean = y_train_fold.mean()

                temp = X_train_fold[[col]].copy()
                temp['target'] = y_train_fold.values

                fold_stats = temp.groupby(col)['target'].agg(['mean', 'count'])
                fold_stats['encoding'] = (
                    (fold_stats['count'] * fold_stats['mean'] + self.smoothing * fold_global_mean) / 
                    (fold_stats['count'] + self.smoothing)
                    )

                fold_encoding = fold_stats['encoding'].to_dict()
                col_oof[val_idx] = X.iloc[val_idx][col].map(fold_encoding).fillna(fold_global_mean)

            encoded_col[f"{col}_target_encoding"] = col_oof

        df = pd.DataFrame(encoded_col, index=X.index)
        X = pd.concat([X,df], axis=1)

        return X

    def transform(self, X):
        check_is_fitted(self, "encoded_map_")
        X = X.copy()
        encoded_col = {
            f"{col}_target_encoding" : X[col].map(self.encoded_map_[col]).fillna(self.global_mean_)
            for col in self.columns
        }
        df = pd.DataFrame(encoded_col, index=X.index)
        X = pd.concat([X,df], axis=1) 
        return X  

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.columns
        return np.array(base + [f"{col}_target_encoding" for col in self.columns])

class MissingIndicator(BaseEstimator, TransformerMixin):
    """ Return Binary Flag 0 or 1 for missing data"""
    def __init__(self, columns = None):
        self.columns = columns

    def fit(self, X, y=None):
        self.fitted_ = True
        self.feature_names_in_ = list(X.columns)
        return self
    
    def transform(self, X):
        check_is_fitted(self, "fitted_")
        X = X.copy()
        missing_col = {
            f"{col}_is_missing" : X[col].isnull().astype(int)
            for col in self.columns
        }
        df = pd.DataFrame(missing_col, index=X.index)
        X = pd.concat([X,df], axis=1)        
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.feature_names_in_
        return np.array(base + [f"{col}_is_missing" for col in self.columns])

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps outlier at specified percentiles"""
    def __init__(self, columns=None, lower_percentile=0.01, upper_percentile=0.99):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile


    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.columns:
            self.lower_bounds_[col] = X[col].quantile(self.lower_percentile)
            self.upper_bounds_[col] = X[col].quantile(self.upper_percentile)
        return self
    
    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].clip(
                lower=self.lower_bounds_[col], 
                upper=self.upper_bounds_[col]
            )
        return X
    
    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.columns
        return np.array(base)

class DropColumns(BaseEstimator, TransformerMixin):
    """ Drops named columns from a dataframe. """
    def __init__(self, columns=None):
        self.columns=columns

    def fit(self, X, y=None):
        self.fitted_=True
        return self
    
    def transform(self, X):
        check_is_fitted(self, "fitted_")
        return X.drop(columns=self.columns, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.array([c for c in input_features if c not in self.columns])
            



