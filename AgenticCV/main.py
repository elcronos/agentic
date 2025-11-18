import os
from pathlib import Path
from typing import Literal

import dspy
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from dspy.evaluate import answer_exact_match
from dspy import Evaluate

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------
# Step 0: Load dataset and build dataframe
# --------------------------------------------------
# Use the script download_dataset.py first to download images
# using capsule/test set here since train set only has "good" images
DATASET = "YOUR_PATH/.cache/kagglehub/datasets/ipythonx/mvtec-ad/versions/2/capsule/test"
DATASET_PATH = Path(DATASET)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_image_dataframe(dataset_path: Path, output_csv: str | None = None) -> pd.DataFrame:
	"""
	Walks a dataset folder where each subfolder is a label and contains images.

	Args:
		dataset_path: Root folder of the dataset.
		output_csv: If provided, saves the resulting DataFrame to this CSV path.

	Returns:
		pd.DataFrame with columns:
			- path: relative path like 'label/image.png'
			- label: subfolder name used as class label
	"""
	rows = []

	for label_dir in dataset_path.iterdir():
		if not label_dir.is_dir():
			continue

		label = label_dir.name

		for img_path in label_dir.iterdir():
			if not img_path.is_file():
				continue

			if img_path.suffix.lower() not in IMAGE_EXTS:
				continue

			rel_path = img_path.relative_to(dataset_path).as_posix()

			rows.append({
				"path": rel_path,
				"label": label,
			})

	df = pd.DataFrame(rows)

	if output_csv is not None:
		df.to_csv(output_csv, index=False)

	return df


# Generate DataFrame
full_df = build_image_dataframe(DATASET_PATH, output_csv="dataset/data.csv")

# --------------------------------------------------
# Build DSPy examples
# --------------------------------------------------
examples = [
	dspy.Example(
		image_path=str(DATASET_PATH / row["path"]),  # store as string, not Path
		answer=row["label"],
	).with_inputs("image_path")
	for _, row in full_df.iterrows()
]

labels = [ex.answer for ex in examples]
print("LABELS:", tuple(labels))

# Stratified 80/20 split based on the labels
train_examples, test_examples = train_test_split(
	examples,
	test_size=0.2,
	stratify=labels,
	random_state=42,
)

# --------------------------------------------------
# DSPy Signature & Module
# --------------------------------------------------
class AnomalyDetectionCapsuleSignature(dspy.Signature):
	#"""
	#Given an image of a capsule, identify any signs of defects. Classify the
	#capsule into one of the following categories: crack, scratch, poke,
	#faulty_imprint, squeeze, or good (if no issues detected in the capsule).
	#"""
	"""
	Examine the provided image of a capsule carefully. Analyze the image to
	detect any visible defects and classify the capsule into one of the
	categories: crack, scratch, poke, faulty_imprint, squeeze, or good.
	Provide a detailed reasoning of your analysis, explaining each step that
	leads to your final classification.
	"""
	image: dspy.Image = dspy.InputField(desc="path to an image to be classified")
	answer: str = dspy.OutputField(desc="classification result")


class AnomalyDetector(dspy.Module):
	def __init__(self):
		super().__init__()
		self.classify = dspy.ChainOfThought(AnomalyDetectionCapsuleSignature)

	def forward(self, image_path: str):
		# Convert path (string) into a DSPy Image
		image = dspy.Image(image_path)
		return self.classify(image=image)

# --------------------------------------------------
# Configure DSPy LM (example – change to your provider/model)
# --------------------------------------------------
dspy.configure(lm=dspy.LM(model="openai/gpt-4o"))

model = AnomalyDetector()
evaluator = Evaluate(devset=test_examples, metric=answer_exact_match)

result = evaluator(model)
print(result)  # e.g. EvaluationResult(score=..., results=<list ...>)
print(f"Baseline exact match: {result.score:.2f}")

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
	metric=answer_exact_match,
	auto="light",
	max_bootstrapped_demos=0,
	max_labeled_demos=0,
	num_threads=8
)

compiled_model = optimizer.compile(
	model,
	trainset=train_examples[:80]
)
# Save optimize program for future use
compiled_model.save(f"output/miprov2_optimized_best.json")

result = evaluator(compiled_model)
print(f"Improved model exact match: {result.score:.2f}")
