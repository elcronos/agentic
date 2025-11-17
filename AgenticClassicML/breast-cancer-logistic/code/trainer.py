import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load training and validation data
train_data = pd.read_parquet("dataset_0_transformed_train.parquet")
val_data = pd.read_parquet("dataset_0_transformed_val.parquet")

# Define features and target variable
X_train = train_data.drop(columns=["diagnosis"])
y_train = train_data["diagnosis"]
X_val = val_data.drop(columns=["diagnosis"])
y_val = val_data["diagnosis"]

# Preprocessing: Define categorical and numerical features
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)

# Create a pipeline that first preprocesses the data and then trains the model
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train the model using the training data
pipeline.fit(X_train, y_train)

# Evaluate the model using the validation data
y_val_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

# Save the model to disk
joblib.dump(pipeline, "linear_regression_model.joblib")

# Print the evaluation metric
print(f"accuracy: {accuracy:.4f}")
