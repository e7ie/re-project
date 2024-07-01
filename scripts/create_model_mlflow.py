import json
import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn import model_selection, neighbors, pipeline, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

SALES_PATH = "data/kc_house_data.csv"  
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containing two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pd.read_csv(sales_path,
                       usecols=sales_column_selection,
                       dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path,
                               dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    
    y = merged_data.pop('price')
    x = merged_data

    return x, y

def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor())

    with mlflow.start_run():
        model.fit(x_train, y_train)

        # Evaluate the model
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

       #Output directory
        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        # Save the model and feature names
        with open(output_dir / "model.pkl", 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(output_dir / "model_features.json", 'w') as features_file:
            json.dump(list(x_train.columns), features_file)

if __name__ == "__main__":
    main()
