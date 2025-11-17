
from typing import List
import joblib
import pandas as pd
from plexe.core.interfaces.predictor import Predictor
from plexe.internal.models.entities.artifact import Artifact

class PredictorImplementation(Predictor):
    def __init__(self, artifacts: List[Artifact]):
        """
        Instantiates the predictor using the provided model artifacts.
        :param artifacts: list of BinaryIO artifacts
        """
        artifact = self._get_artifact("linear_regression_model.joblib", artifacts)
        with artifact.get_as_handle() as binary_io:
            self.model = joblib.load(binary_io)

    def predict(self, inputs: dict) -> dict:
        """
        Given an input conforming to the input schema, return the model's prediction
        as a dict conforming to the output schema.
        """
        preprocessed_input = self._preprocess_input(inputs)
        prediction = self.model.predict(preprocessed_input)
        return self._postprocess_output(prediction)

    def _preprocess_input(self, inputs: dict):
        """
        Map the input data from a dict to a DataFrame with the required columns for the underlying model.
        """
        # List of required columns as per the model training
        required_columns = [
            'perimeter_worst', 'radius_worst', 'concave_points_worst',
            'texture_worst', 'concave points_mean', 'compactness_mean', 
            'perimeter_mean', 'fractal_dimension_worst', 'poly_1', 'poly_2', 
            'poly_3', 'smoothness_se', 'perimeter_se', 'poly_6', 
            'poly_8', 'smoothness_mean', 'fractal_dimension_mean', 
            'texture_mean', 'concavity_mean', 'compactness_se', 
            'radius_se', 'smoothness_worst', 'area_mean', 'symmetry_mean', 
            'concavity_se', 'poly_4', 'radius_mean', 'texture_se', 
            'concave points_se', 'compactness_worst', 'poly_5', 
            'symmetry_se', 'concavity_worst', 'symmetry_worst', 
            'poly_0', 'area_worst', 'area_se', 'fractal_dimension_se', 
            'poly_7'
        ]
        
        # Prepare the input data with defaults for missing columns
        processed_data = {key: inputs.get(key, 0) for key in required_columns}
        return pd.DataFrame([processed_data])

    def _postprocess_output(self, outputs) -> dict:
        """
        Map the output from the underlying model to a dict compliant with the output schema.
        """
        # Assuming the prediction is binary, we need to interpret it as strings
        diagnosis_mapping = {0: "Benign", 1: "Malignant"}
        return {"diagnosis": diagnosis_mapping.get(outputs[0], "Unknown")}  # Map the output to a string.

    @staticmethod
    def _get_artifact(name: str, artifacts: List[Artifact]) -> Artifact:
        """
        Given the name of a binary artifact, return the corresponding artifact from the list.
        """
        for artifact in artifacts:
            if artifact.name == name:
                return artifact
        raise ValueError(f"Artifact {name} not found in the provided artifacts.")
