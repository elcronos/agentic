import os
import pandas as pd
from dotenv import load_dotenv

import plexe
from plexe import ModelBuilder
from plexe.internal.common.provider import ProviderConfig

# PATCH: fix smolagents import check
import smolagents.local_python_executor as lpe
def _no_check(self):
	return None
lpe.LocalPythonExecutor._check_authorized_imports_are_installed = _no_check


# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------
# Step 0: Load and split dataset into train / test
# --------------------------------------------------
full_df = pd.read_csv("dataset/data.csv")

# Shuffle
full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 80% train, 20% test
split_idx = int(0.8 * len(full_df))
train_df = full_df.iloc[:split_idx].copy()
test_df = full_df.iloc[split_idx:].copy()

# (Optional) save to disk
train_df.to_csv("dataset/train.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

# Step 1: Define the model
# Note: for conciseness we leave the input schema empty and let plexe infer it
model = ModelBuilder(
    provider=ProviderConfig(
        default_provider="openai/gpt-4o",
		# You can specify other providers for different agents
        # orchestrator_provider="anthropic/claude-sonnet-4-20250514",
        # research_provider="openai/gpt-4o",
        # engineer_provider="anthropic/claude-3-7-sonnet-20250219",
        # ops_provider="anthropic/claude-3-7-sonnet-20250219",
        # tool_provider="openai/gpt-4o",
    ),
    verbose=False,
)

# Step 2: Build the model using the training dataset
m = model.build(
    datasets=[train_df],
    intent=(
        """Diagnose from the characteristics of the cell nuclei present in the
		 dataset as M = malignant  or B = Benign the final price of each home.
		 Use only linear regression and decision tree models, no ensembling.
		 The models must be extremely simple and quickly trainable on extremely
		 constrained hardware."""
    ),
    output_schema={
        "diagnosis": str,
    },
    max_iterations=2,
    timeout=1800,  # 30 minute timeout
    run_timeout=180
)

# Step 3: Save the model
plexe.save_model(m, "breast-cancer.tar.gz")

# Step 4: Run a prediction on the built model
test_df = test_df.sample(20)
predictions = pd.DataFrame.from_records([m.predict(x) for x in test_df.to_dict(orient="records")])

# Step 5: print a sample of predictions
print(predictions)

# Step 6: Print model description
description = m.describe()
print(description.as_text())
