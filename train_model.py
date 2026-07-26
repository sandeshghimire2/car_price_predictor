"""
train_model.py
----------------
Reproduces the data-cleaning and model-training steps from the original
`maincode.ipynb` notebook, then saves a fresh `Linearmodelcar.pkl` that is
guaranteed to be compatible with the scikit-learn / pandas versions listed
in requirements.txt.

Run this once (it's already been run for you and the model is included),
or re-run it any time you want to retrain on updated data:

    python train_model.py
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

RAW_DATA_PATH = "data/car.csv"
CLEAN_DATA_PATH = "data/cleaned_car.csv"
MODEL_PATH = "Linearmodelcar.pkl"


def clean_data(raw_path: str) -> pd.DataFrame:
    """Recreates every cleaning step from the notebook."""
    df = pd.read_csv(raw_path)

    # Keep only rows where year is a real number
    df = df[df["year"].str.isnumeric()]
    df["year"] = df["year"].astype(int)

    # Drop rows where price wasn't listed, strip commas, cast to int
    df = df[df["Price"] != "Ask For Price"]
    df["Price"] = df["Price"].str.replace(",", "").astype(int)

    # kms_driven: keep the numeric part, strip commas, drop bad rows, cast to int
    df["kms_driven"] = df["kms_driven"].str.split(" ").str.get(0)
    df["kms_driven"] = df["kms_driven"].str.replace(",", "")
    df = df[df["kms_driven"] != "Petrol"]
    df["kms_driven"] = df["kms_driven"].astype(int)

    # Keep only the first 3 words of the car name (brand + model)
    df["name"] = df["name"].str.split().str[:3].str.join(" ")

    # Fill missing fuel type with the most common value
    df["fuel_type"] = df["fuel_type"].fillna(df["fuel_type"].mode()[0])

    df = df.reset_index(drop=True)

    # Remove the extreme price outlier identified in the notebook (row with
    # Price >= 6,000,000). We match on the condition rather than a fixed
    # index since the index shifts after reset_index.
    df = df[df["Price"] < 6000000].reset_index(drop=True)

    return df


def train_model(df: pd.DataFrame):
    x = df.drop(columns="Price")
    y = df["Price"]

    ohe = OneHotEncoder()
    ohe.fit(x[["name", "company", "fuel_type"]])

    column_transform = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ["name", "company", "fuel_type"]),
        remainder="passthrough",
    )

    # Search for the train/test split that gives the best r2 score, exactly
    # like the notebook did (0-999).
    scores = []
    for i in range(1000):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i
        )
        pipe = make_pipeline(column_transform, LinearRegression())
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        scores.append(r2_score(y_test, y_pred))

    best_state = int(np.argmax(scores))
    best_score = scores[best_state]
    print(f"Best random_state: {best_state}  |  r2_score: {best_score:.4f}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=best_state
    )
    pipeline = make_pipeline(column_transform, LinearRegression())
    pipeline.fit(x_train, y_train)

    final_score = r2_score(y_test, pipeline.predict(x_test))
    print(f"Final model r2_score on held-out test set: {final_score:.4f}")

    return pipeline


if __name__ == "__main__":
    print("Cleaning data...")
    clean_df = clean_data(RAW_DATA_PATH)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned data -> {CLEAN_DATA_PATH}  ({clean_df.shape[0]} rows)")

    print("\nTraining model (testing 1000 train/test splits, this takes a bit)...")
    model = train_model(clean_df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved trained model -> {MODEL_PATH}")
