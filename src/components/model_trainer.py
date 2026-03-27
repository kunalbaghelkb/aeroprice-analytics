import os
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from src.logger import logging
from src.constants import MODEL_DIR

class ModelTrainer:
    def __init__(self):
        # Add DagsHub's URI
        # mlflow.set_tracking_uri("YOUR_DAGSHUB_MLFLOW_URI")
        # mlflow.set_experiment("Used_Car_Pricing_XGBoost")
        pass

    def evaluate_model(self, true, predicted):
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2 = r2_score(true, predicted)
        return rmse, r2

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting Model Training Phase with XGBoost...")

            # Best parameters we found from RandomizedSearchCV in our Notebook
            best_params = {
                'n_estimators': 350,
                'learning_rate': 0.1,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }

            model = XGBRegressor(**best_params)

            # MLflow Tracking Start
            # with mlflow.start_run():
            logging.info("Training the model...")
            model.fit(X_train, y_train)

            logging.info("Predicting on Test Data...")
            y_pred = model.predict(X_test)

            rmse, r2 = self.evaluate_model(y_test, y_pred)
            logging.info(f"Model Performance - RMSE: {rmse}, R2: {r2}")

            # Logging params and metrics to MLflow
            # mlflow.log_params(best_params)
            # mlflow.log_metric("rmse", rmse)
            # mlflow.log_metric("r2", r2)
            # mlflow.xgboost.log_model(model, "xgboost_model")
            # MLflow Tracking End

            # Save the model locally in models/ directory
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, "xgboost_pricing_master.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model successfully saved at {model_path}")

            return rmse, r2

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise e