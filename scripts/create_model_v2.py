# scripts/create_model_v2.py
import json
import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn import model_selection, linear_model, ensemble, pipeline, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
    'sqft_lot15'
]
OUTPUT_DIR = "model/v2"

def load_data(sales_path: str, demographics_path: str, sales_column_selection: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(sales_path, usecols=sales_column_selection, dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})
    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    x = merged_data
    return x, y

def main():
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

    # Different models to try
    models = {
        "Linear Regression": linear_model.LinearRegression(),
        "Random Forest": ensemble.RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": ensemble.GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline_model = pipeline.make_pipeline(preprocessing.RobustScaler(), model)
            pipeline_model.fit(x_train, y_train)
            predictions = pipeline_model.predict(x_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            print(f"{model_name} - Mean Squared Error: {mse}")
            print(f"{model_name} - R^2 Score: {r2}")
            mlflow.log_param("model", model_name)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(pipeline_model, "model")
            output_dir = pathlib.Path(OUTPUT_DIR) / model_name.replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "model.pkl", 'wb') as model_file:
                pickle.dump(pipeline_model, model_file)
            with open(output_dir / "model_features.json", 'w') as features_file:
                json.dump(list(x_train.columns), features_file)

if __name__ == "__main__":
    main()
