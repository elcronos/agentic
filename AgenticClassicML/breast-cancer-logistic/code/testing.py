
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Preparing the input for predictions
input_data = test_df.drop('diagnosis', axis=1).to_dict(orient="records")
predictions = predictor.predict({"instances": input_data})

# Extracting true values and predicted values
y_true = test_df['diagnosis'].tolist()  # True labels
y_pred_numeric = [1 if pred['diagnosis'] == 'Malignant' else 0 for pred in [predictions]] * len(y_true)  # Numeric predictions
y_true_numeric = [1 if label == 'Malignant' else 0 for label in y_true]  # Numeric true labels

# Calculating performance metrics
accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
f1 = f1_score(y_true_numeric, y_pred_numeric, average='weighted')
precision = precision_score(y_true_numeric, y_pred_numeric, average='weighted')
recall = recall_score(y_true_numeric, y_pred_numeric, average='weighted')

performance_summary = {
    "Accuracy": accuracy,
    "F1 Score": f1,
    "Precision": precision,
    "Recall": recall
}
