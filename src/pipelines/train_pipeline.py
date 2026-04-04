import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.components.data_loader import VehicleDataLoader
from src.components.preprocessor import VehicleDataPreprocessor
from src.components.model_trainer import ModelTrainer
from src.constants import RAW_DATA_PATH, MODEL_DIR
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_loader = VehicleDataLoader(file_path=RAW_DATA_PATH)
        self.preprocessor = VehicleDataPreprocessor()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting the Master Training Pipeline...")

            # Step 1: Load Data
            df = self.data_loader.load_data()
            
            # Step 1a: OUTLIER REMOVAL
            from src.constants import LOWER_PRICE_BOUND, UPPER_PRICE_BOUND
            
            logging.info(f"Removing price outliers (Keeping between ${LOWER_PRICE_BOUND} and ${UPPER_PRICE_BOUND})...")
            df = df[(df['price'] >= LOWER_PRICE_BOUND) & (df['price'] <= UPPER_PRICE_BOUND)].copy()
            logging.info(f"Data shape after outlier removal: {df.shape}")

            # Step 2: Train-Test Split ON FULL DATA (Prevents data leakage completely)
            logging.info("Performing Train-Test Split...")
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info("Data Split complete. Moving to Preprocessing...")

            # Step 3: Preprocess Data
            # fit_transform needs the full dataframe because it uses 'price' for Target Encoding
            df_train_processed = self.preprocessor.fit_transform(df_train)

            # Now safely separate X (Features) and y (Target) for training
            y_train = df_train_processed['price']
            X_train_processed = df_train_processed.drop(columns=['price'])

            # For test data, we mimic the real-world API behavior (passing only features, no price)
            y_test = df_test['price']
            X_test = df_test.drop(columns=['price'])
            X_test_processed = self.preprocessor.transform(X_test)

            # Step 4: Scale Data
            logging.info("Scaling features...")
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_processed)
            X_test_scaled = scaler.transform(X_test_processed)
            
            # Step 5: Save All Artifacts for Production API
            logging.info("Saving artifacts for API inference...")
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # 5a. Save Scaler
            joblib.dump(scaler, os.path.join(MODEL_DIR, 'robust_scaler.pkl'))
            
            # 5b. Save Preprocessor state
            joblib.dump(self.preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
            
            # 5c. Save exact column blueprint
            training_columns = X_train_processed.columns.tolist()
            joblib.dump(training_columns, os.path.join(MODEL_DIR, 'training_columns.pkl'))
            
            logging.info("All artifacts (Scaler, Preprocessor, Columns) saved successfully!")

            # Step 6: Train Model and Track
            logging.info("Initiating Model Training...")
            r2_score = self.model_trainer.initiate_model_trainer(
                X_train_scaled, y_train, X_test_scaled, y_test
            )

            logging.info(f"Training Pipeline Complete! Final Model R2 Score: {r2_score}")

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise e

# Run Training pipeline
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()