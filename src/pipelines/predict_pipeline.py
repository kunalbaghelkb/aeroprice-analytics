import os
import joblib
import pandas as pd
import numpy as np
from src.logger import logging
from src.constants import MODEL_DIR

class PredictPipeline:
    def __init__(self):
        try:
            logging.info("Loading Prediction Artifacts...")
            self.model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_pricing_master.pkl'))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, 'robust_scaler.pkl'))
            self.training_cols = joblib.load(os.path.join(MODEL_DIR, 'training_columns.pkl'))
            self.preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
            logging.info("Artifacts loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load prediction artifacts: {e}")
            raise e

    def predict(self, input_dataframe):
        try:
            logging.info(f"Starting Prediction Pipeline for {len(input_dataframe)} rows...")
            
            # Step 1: Preprocess the raw input (OHE, Target Encoding, Imputation)
            processed_df = self.preprocessor.transform(input_dataframe)
            
            # Step 2: Ensure column mismatch doesn't happen (Reindexing)
            processed_df = processed_df.reindex(columns=self.training_cols, fill_value=0)
            
            # Step 3: Scale the data
            scaled_array = self.scaler.transform(processed_df)
            scaled_df = pd.DataFrame(scaled_array, columns=self.training_cols)
            
            # Step 4: Predict for all rows
            predictions = self.model.predict(scaled_df)
            
            # Post-process and floor results to $500 min
            results = [max(500.0, float(p)) for p in predictions]
            
            logging.info(f"Batch prediction successful for {len(results)} items.")
            
            # Return single float for single row (backward compatibility)
            if len(results) == 1:
                return results[0]
            
            return results
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
