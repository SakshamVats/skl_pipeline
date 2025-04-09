import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)

def get_model(model_name, params):
    models = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svc": SVC
        }
    return models[model_name](**params)

def load_data():
    df = pd.read_csv("data/adult_income.csv")
    df.replace("?", np.nan, inplace=True)
    return df

def build_pipeline(numeric_features, categorical_features, model_instance):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model_instance)
    ])

    return pipeline

def run_pipeline():
    config = load_config()
    model_name = config["model"]
    model_params = config["hyperparameters"]

    model_instance = get_model(model_name, model_params)

    df = load_data()

    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    pipeline = build_pipeline(numeric_features, categorical_features, model_instance)

    # Cross-validation on training data
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

    # Train on full training set
    pipeline.fit(X_train, y_train)

    # Save trained pipeline to joblib file
    joblib.dump(pipeline, "config/model.pkl")
    print("Model saved to config/model.pkl!")

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot
    plt.bar(["Cross-Validation", "Test"], [cv_scores.mean(), test_accuracy], color=["skyblue", "salmon"])
    plt.ylim(0.5, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Performance")
    plt.show()

if __name__ == "__main__":
    run_pipeline()
