import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.logger import logging
from src import constants

class VehicleDataPreprocessor:
    def __init__(self):
        # State variables to store learned parameters for API inference
        self.median_odometers_per_year = {}
        self.global_odometer_median = 0
        self.global_price_mean = 0
        self.target_encoding_maps = {col: {} for col in constants.HIGH_CARD_COLS}
        self.training_columns = None

    def fit_transform(self, df):
        """Used strictly during TRAINING. Learns parameters and applies K-Fold encoding."""
        logging.info("Starting fit_transform pipeline for Training...")
        df = df.copy()

        # 1. Drop useless columns
        df = df.drop(columns=[col for col in constants.COLS_TO_DROP_INITIAL + constants.COLS_TO_DROP_FINAL if col in df.columns])
        df = df.dropna(subset=['year', 'manufacturer'])
        
        # 2. Imputation learning
        self.global_price_mean = df['price'].mean()
        
        df['year'] = df['year'].astype('float32')
        df['odometer'] = df['odometer'].astype('float32')
        self.median_odometers_per_year = df.groupby('year')['odometer'].median().to_dict()
        self.global_odometer_median = df['odometer'].median()
        
        df['odometer'] = df['odometer'].fillna(df['year'].map(self.median_odometers_per_year))
        df['odometer'] = df['odometer'].fillna(self.global_odometer_median)

        for col in constants.CATEGORICAL_MISSING_COLS:
            if 'unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('unknown')
            df[col] = df[col].fillna('unknown')

        for col in constants.MINOR_CATS:
            df[col] = df[col].fillna(df[col].mode()[0])

        # 3. Ordinal Encoding
        df['condition'] = df['condition'].map(constants.COND_MAP).astype('int8')
        df['cylinders'] = df['cylinders'].map(constants.CYL_MAP).astype('int8')
        df['size'] = df['size'].map(constants.SIZE_MAP).astype('int8')

        # 4. Target Encoding (K-Fold for training, global means saved for API)
        for col in constants.HIGH_CARD_COLS:
            df[f"{col}_encoded"] = np.nan
            df[col] = df[col].fillna('unknown')
            
            # Save global category means for API inference later
            self.target_encoding_maps[col] = df.groupby(col)['price'].mean().to_dict()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(df):
            X_train_fold = df.iloc[train_idx]
            X_val_fold = df.iloc[val_idx]
            
            for col in constants.HIGH_CARD_COLS:
                fold_means = X_train_fold.groupby(col)['price'].mean()
                df.iloc[val_idx, df.columns.get_loc(f"{col}_encoded")] = X_val_fold[col].map(fold_means)

        for col in constants.HIGH_CARD_COLS:
            encoded_name = f"{col}_encoded"
            df[encoded_name] = df[encoded_name].fillna(self.global_price_mean)
            df = df.drop(columns=[col]).rename(columns={encoded_name: col})
            df[col] = df[col].astype('float32')

        # 5. One-Hot Encoding
        df = pd.get_dummies(df, columns=constants.OHE_COLS, drop_first=True, dtype='int8')
        
        # Save exact column order for API mapping
        self.training_columns = df.drop(columns=['price']).columns.tolist()
        
        logging.info("fit_transform complete.")
        return df

    def transform(self, df):
        """Used strictly during INFERENCE (API). Uses learned parameters, NO K-Fold."""
        logging.info("Starting transform pipeline for Inference/API...")
        df = df.copy()

        # Drop initial columns if they exist in API request
        df = df.drop(columns=[col for col in constants.COLS_TO_DROP_INITIAL + constants.COLS_TO_DROP_FINAL if col in df.columns], errors='ignore')

        # Impute
        df['year'] = df['year'].astype('float32')
        df['odometer'] = df['odometer'].astype('float32')
        df['odometer'] = df['odometer'].fillna(df['year'].map(self.median_odometers_per_year))
        df['odometer'] = df['odometer'].fillna(self.global_odometer_median)

        # Categorical handling
        for col in constants.CATEGORICAL_MISSING_COLS:
            # Note: For API, we might just have strings, so we convert to string then fill
            df[col] = df[col].astype(str).replace('nan', 'unknown').fillna('unknown')

        # Ordinal Encode
        df['condition'] = df['condition'].map(constants.COND_MAP).fillna(0).astype('int8')
        df['cylinders'] = df['cylinders'].map(constants.CYL_MAP).fillna(0).astype('int8')
        df['size'] = df['size'].map(constants.SIZE_MAP).fillna(0).astype('int8')

        # Target Encode (Using saved maps, NOT K-Fold)
        for col in constants.HIGH_CARD_COLS:
            df[col] = df[col].map(self.target_encoding_maps[col]).fillna(self.global_price_mean).astype('float32')

        # One-Hot Encoding
        df = pd.get_dummies(df, columns=constants.OHE_COLS, drop_first=True, dtype='int8')

        # Reindex to match training columns EXACTLY (Fills missing dummy columns with 0)
        df = df.reindex(columns=self.training_columns, fill_value=0)
        
        logging.info("transform complete. Ready for prediction.")
        return df