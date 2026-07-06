import os
import re
import sys
import logging
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import pickle
import time
import argparse
import json

# Optional progress bar
try:
	from tqdm import tqdm
	tqdm_available = True
except Exception:
	tqdm_available = False


def setup_logging():
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(message)s",
		level=logging.INFO,
		stream=sys.stdout,
	)


def download_nltk_resources():
	resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
	for res in resources:
		try:
			logging.info(f"Downloading NLTK resource: {res}")
			nltk.download(res, quiet=True)
		except Exception as e:
			logging.warning(f"Could not download {res}: {e}")


def load_dataset(csv_path: str) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Dataset not found at: {csv_path}")
	logging.info(f"Loading dataset from {csv_path}")
	df = pd.read_csv(csv_path)
	logging.info(f"Loaded dataframe with shape: {df.shape}")
	return df


def perform_eda(df: pd.DataFrame, plots_dir: str = "eda_plots") -> None:
	os.makedirs(plots_dir, exist_ok=True)

	logging.info("Head of dataset:\n%s", df.head().to_string(index=False))
	logging.info("Columns: %s", df.columns.tolist())
	logging.info("Dtypes:\n%s", df.dtypes)

	# Class distribution
	if "label" in df.columns:
		counts = df["label"].value_counts()
		pct = df["label"].value_counts(normalize=True) * 100
		logging.info("Class counts:\n%s", counts.to_string())
		logging.info("Class distribution (percent):\n%s", pct.round(2).to_string())

		plt.figure(figsize=(6, 4))
		sns.countplot(x="label", data=df)
		plt.title("Class distribution: ham vs spam")
		plt.tight_layout()
		figpath = os.path.join(plots_dir, "class_distribution.png")
		plt.savefig(figpath)
		plt.close()
		logging.info("Saved class distribution plot to %s", figpath)

	# Missing values
	missing = df.isnull().sum()
	missing_pct = (missing / len(df)) * 100
	logging.info("Missing values (counts):\n%s", missing.to_string())
	logging.info("Missing values (percent):\n%s", missing_pct.round(3).to_string())

	# Text length statistics
	if "text" in df.columns:
		df["text_length"] = df["text"].astype(str).str.len()
		df["word_count"] = df["text"].astype(str).str.split().apply(len)

		logging.info("Text length stats:\n%s", df["text_length"].describe().to_string())
		logging.info("Word count stats:\n%s", df["word_count"].describe().to_string())

		# Boxplot of text lengths by label
		plt.figure(figsize=(8, 5))
		sns.boxplot(x="label", y="text_length", data=df)
		plt.title("Text length by class")
		plt.tight_layout()
		figpath = os.path.join(plots_dir, "text_length_by_label.png")
		plt.savefig(figpath)
		plt.close()
		logging.info("Saved text length boxplot to %s", figpath)

		# Boxplot of word counts
		plt.figure(figsize=(8, 5))
		sns.boxplot(x="label", y="word_count", data=df)
		plt.title("Word count by class")
		plt.tight_layout()
		figpath = os.path.join(plots_dir, "word_count_by_label.png")
		plt.savefig(figpath)
		plt.close()
		logging.info("Saved word count boxplot to %s", figpath)

	# Duplicates
	dup_count = df.duplicated().sum()
	logging.info("Duplicate rows: %d", int(dup_count))

	# Character composition insights (sample)
	if "text" in df.columns:
		sample_text = df["text"].dropna().astype(str).iloc[0:200].str.cat(sep=" ")
		uppercase = sum(1 for c in sample_text if c.isupper())
		digits = sum(1 for c in sample_text if c.isdigit())
		specials = sum(1 for c in sample_text if not c.isalnum() and not c.isspace())
		logging.info("Sample character composition (first 200 texts combined): uppercase=%d digits=%d special=%d", uppercase, digits, specials)

	# Show sample ham and spam
	if "label" in df.columns:
		for lbl in ["ham", "spam"]:
			sample = df[df["label"] == lbl]
			if not sample.empty:
				logging.info("Sample %s email (first):\n%s", lbl, str(sample["text"].astype(str).iloc[0])[:500])


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words_set: set) -> str:
	try:
		if not isinstance(text, str):
			text = str(text)

		# Lowercase
		text = text.lower()

		# Remove non-letters (keep spaces)
		text = re.sub(r"[^a-zA-Z\s]", " ", text)

		# Collapse whitespace
		text = re.sub(r"\s+", " ", text).strip()

		# Tokenize using a regex to avoid dependency on punkt tokenizer
		# extract alphabetic tokens
		tokens = re.findall(r"\b[a-z]+\b", text)

		# Stopword removal but keep 'not' and 'no'
		filtered = [t for t in tokens if t not in stop_words_set]

		# Lemmatize
		lemmatized = [lemmatizer.lemmatize(t) for t in filtered]

		return " ".join(lemmatized)
	except Exception as e:
		logging.debug("Error preprocessing text: %s", e)
		return ""


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
	logging.info("Preparing stopwords and lemmatizer")
	stop_words = set(stopwords.words("english"))
	# keep 'not' and 'no' which can be important for negation
	for keep in ("not", "no"):
		stop_words.discard(keep)

	lemmatizer = WordNetLemmatizer()

	# Use tqdm if available
	if tqdm_available:
		tqdm.pandas(desc="Preprocessing texts")
		df["cleaned_text"] = df["text"].progress_apply(lambda t: preprocess_text(t, lemmatizer, stop_words))
	else:
		df["cleaned_text"] = df["text"].apply(lambda t: preprocess_text(t, lemmatizer, stop_words))

	# Show before/after samples
	logging.info("Original text (sample): %s", str(df["text"].astype(str).iloc[0])[:300])
	logging.info("Cleaned text (sample): %s", str(df["cleaned_text"].astype(str).iloc[0])[:300])

	return df


def split_and_save(df: pd.DataFrame, out_train: str = "train_data.csv", out_test: str = "test_data.csv") -> None:
	# Validate presence of required columns
	if "cleaned_text" not in df.columns or "label_num" not in df.columns:
		raise ValueError("DataFrame must contain 'cleaned_text' and 'label_num' columns before splitting")

	# Ensure label_num matches label if label present
	if "label" in df.columns:
		# enforce canonical mapping: ham -> 0, spam -> 1
		mapping = {"ham": 0, "spam": 1}
		df["label_norm"] = df["label"].astype(str).str.lower().map(mapping)
		# where mapping failed, fallback to existing label_num
		df["label_num"] = df.apply(lambda r: int(r["label_num"]) if pd.isna(r.get("label_norm")) else int(r["label_norm"]), axis=1)

	# Drop rows with empty cleaned_text as they are unusable
	empty_text_count = df["cleaned_text"].astype(str).str.strip().replace("", pd.NA).isna().sum()
	if empty_text_count > 0:
		logging.warning("Found %d rows with empty 'cleaned_text'. These will be dropped before splitting.", int(empty_text_count))
		df = df[df["cleaned_text"].astype(str).str.strip() != ""].reset_index(drop=True)

	if df.empty:
		raise ValueError("No valid rows with non-empty 'cleaned_text' available to split.")

	# Use the DataFrame indices to split so we can recover all original columns reliably
	indices = df.index.values
	y = df["label_num"].values

	logging.info("Splitting dataset (80/20) with stratify and random_state=42 using indices")
	idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

	train_df = df.loc[idx_train, ["cleaned_text", "label", "label_num"]].reset_index(drop=True)
	test_df = df.loc[idx_test, ["cleaned_text", "label", "label_num"]].reset_index(drop=True)

	logging.info("Train size: %d, Test size: %d", len(train_df), len(test_df))

	# Final validation before saving
	for name, frame in ((out_train, train_df), (out_test, test_df)):
		if frame["cleaned_text"].astype(str).str.strip().replace("", pd.NA).isna().all():
			logging.error("After splitting, %s has all empty 'cleaned_text' — aborting save.", name)
			raise ValueError(f"{name} would have empty 'cleaned_text' column")

	train_df.to_csv(out_train, index=False)
	test_df.to_csv(out_test, index=False)

	logging.info("Saved train data to %s", out_train)
	logging.info("Saved test data to %s", out_test)


def validate_preprocessed_data(csv_path: str) -> bool:
	"""Validate a preprocessed CSV contains usable cleaned_text and correct labels.

	Returns True if valid, False otherwise.
	"""
	try:
		if not os.path.exists(csv_path):
			logging.warning("Validation failed: %s does not exist.", csv_path)
			return False
		df = pd.read_csv(csv_path)
		if "cleaned_text" not in df.columns or "label_num" not in df.columns:
			logging.warning("Validation failed: required columns missing in %s", csv_path)
			return False
		total = len(df)
		empty = df["cleaned_text"].astype(str).str.strip().replace("", pd.NA).isna().sum()
		if total == 0 or empty == total:
			logging.warning("Validation failed: %s has %d/%d empty cleaned_text", csv_path, int(empty), int(total))
			return False
		if not set(df["label_num"].unique()).issubset({0, 1}):
			logging.warning("Validation failed: label_num contains values other than 0/1 in %s", csv_path)
			return False
		logging.info("Validation OK for %s: rows=%d, empty_text=%d", csv_path, int(total), int(empty))
		return True
	except Exception as e:
		logging.exception("Error during validation of %s: %s", csv_path, e)
		return False


def extract_tfidf_features(train_csv_path: str = "train_data.csv",
						   test_csv_path: str = "test_data.csv",
						   vectorizer_output_path: str = "vectorizer.pkl",
						   max_features: int = 3000):
	"""Load train/test CSVs, fit a TfidfVectorizer on train, transform both and save the vectorizer.

	Returns: X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
	"""
	logging.info("Loading preprocessed CSVs for TF-IDF extraction")
	if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
		raise FileNotFoundError("Train/test CSVs not found. Run preprocessing phase first.")

	train_df = pd.read_csv(train_csv_path)
	test_df = pd.read_csv(test_csv_path)

	# Basic validation
	if "cleaned_text" not in train_df.columns or "cleaned_text" not in test_df.columns:
		raise ValueError("Both train and test must contain 'cleaned_text' column")

	X_train_text = train_df["cleaned_text"].fillna("").astype(str).tolist()
	X_test_text = test_df["cleaned_text"].fillna("").astype(str).tolist()
	y_train = train_df["label_num"].astype(int).values
	y_test = test_df["label_num"].astype(int).values

	logging.info("Initializing TfidfVectorizer(max_features=%d, ngram_range=(1,2), min_df=2, max_df=0.95)", max_features)
	vectorizer = TfidfVectorizer(
		max_features=max_features,
		ngram_range=(1, 2),
		min_df=2,
		max_df=0.95,
		sublinear_tf=True,
		lowercase=False,
		stop_words=None,
	)

	logging.info("Fitting vectorizer on training data")
	X_train_tfidf = vectorizer.fit_transform(X_train_text)
	X_test_tfidf = vectorizer.transform(X_test_text)

	logging.info("Train TF-IDF shape: %s", str(X_train_tfidf.shape))
	logging.info("Test TF-IDF shape: %s", str(X_test_tfidf.shape))

	# Sparsity
	sparsity = 100.0 * (1.0 - (X_train_tfidf.nnz / float(X_train_tfidf.shape[0] * X_train_tfidf.shape[1])))
	logging.info("Train matrix sparsity: %.3f%%", sparsity)

	# Save vectorizer
	try:
		with open(vectorizer_output_path, "wb") as f:
			pickle.dump(vectorizer, f)
		logging.info("Saved vectorizer to %s", vectorizer_output_path)
	except Exception as e:
		logging.exception("Failed to save vectorizer: %s", e)

	# Quick verification load
	try:
		with open(vectorizer_output_path, "rb") as f:
			_ = pickle.load(f)
		logging.info("Verified vectorizer save/load OK")
	except Exception as e:
		logging.warning("Could not verify vectorizer load: %s", e)

	return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_and_compare_models(X_train, y_train, X_test, y_test, save_csv: str = "model_performance.csv"):
	"""Train a set of models and compare on test set. Returns a results DataFrame and dict of fitted models."""
	models = {
		"Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
		"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver="liblinear"),
		"Linear SVM": LinearSVC(max_iter=10000, random_state=42, dual=False),
		"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
		"Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
	}

	results = []
	fitted_models = {}

	total_models = len(models)
	for idx, (name, model) in enumerate(models.items(), start=1):
		logging.info("Training model %d/%d: %s", idx, total_models, name)
		start = time.time()
		try:
			# Some tree-based models may not accept sparse input; try directly and fallback to dense
			try:
				model.fit(X_train, y_train)
			except Exception:
				X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.array(X_train)
				model.fit(X_train_dense, y_train)

			train_time = time.time() - start
			fitted_models[name] = model

			# Predict
			try:
				y_pred = model.predict(X_test)
			except Exception:
				X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else np.array(X_test)
				y_pred = model.predict(X_test_dense)

			acc = accuracy_score(y_test, y_pred)
			prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
			rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
			f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

			results.append({
				"model_name": name,
				"accuracy": acc,
				"precision": prec,
				"recall": rec,
				"f1_score": f1,
				"training_time": round(train_time, 3),
			})

			logging.info("%s - acc: %.4f prec: %.4f rec: %.4f f1: %.4f time: %.2fs", name, acc, prec, rec, f1, train_time)
		except Exception as e:
			logging.exception("Model %s failed: %s", name, e)

	results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
	try:
		results_df.to_csv(save_csv, index=False)
		logging.info("Saved model comparison to %s", save_csv)
	except Exception as e:
		logging.warning("Could not save model comparison CSV: %s", e)

	# Log full table
	logging.info("Model comparison:\n%s", results_df.to_string(index=False))

	return results_df, fitted_models


def select_and_save_best_model(results_df: pd.DataFrame, fitted_models: dict, model_output_path: str = "model.pkl", metadata_path: str = "model_metadata.json", vectorizer=None):
	"""Select best model by f1_score, save model pickle and metadata JSON."""
	if results_df.empty:
		raise ValueError("Empty results_df provided")

	best_row = results_df.iloc[0]
	best_name = best_row["model_name"]
	best_f1 = float(best_row["f1_score"])

	if best_name not in fitted_models:
		raise KeyError(f"Best model '{best_name}' not found in fitted models")

	best_model = fitted_models[best_name]

	# Save model
	try:
		with open(model_output_path, "wb") as f:
			pickle.dump(best_model, f)
		logging.info("Saved best model (%s) to %s", best_name, model_output_path)
	except Exception as e:
		logging.exception("Failed to save best model: %s", e)

	# Verify load
	try:
		with open(model_output_path, "rb") as f:
			_ = pickle.load(f)
		logging.info("Verified model save/load OK")
	except Exception as e:
		logging.warning("Could not verify model load: %s", e)

	# Metadata
	metadata = {
		"model_name": best_name,
		"f1_score": best_f1,
		"accuracy": float(best_row.get("accuracy", 0)),
		"precision": float(best_row.get("precision", 0)),
		"recall": float(best_row.get("recall", 0)),
		"training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
	}
	if vectorizer is not None:
		try:
			metadata["feature_count"] = len(vectorizer.get_feature_names_out())
			metadata["vectorizer_config"] = {
				"max_features": getattr(vectorizer, "max_features", None),
				"ngram_range": getattr(vectorizer, "ngram_range", None),
				"min_df": getattr(vectorizer, "min_df", None),
				"max_df": getattr(vectorizer, "max_df", None),
				"sublinear_tf": getattr(vectorizer, "sublinear_tf", None),
			}
		except Exception:
			pass

	try:
		with open(metadata_path, "w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2)
		logging.info("Saved model metadata to %s", metadata_path)
	except Exception as e:
		logging.warning("Could not save metadata: %s", e)

	return best_model, best_name, metadata
def main():
	setup_logging()

	parser = argparse.ArgumentParser(description="Spam detection pipeline: preprocess, tfidf, train, select")
	parser.add_argument("--phase", choices=["all", "preprocess", "tfidf", "train", "select"], default="all",
						help="Which phase to run")
	parser.add_argument("--rerun-preprocess", action="store_true", help="Force re-running preprocessing even if CSVs exist")
	parser.add_argument("--output-dir", type=str, default=None, help="Directory to save/sync the trained model artifacts")
	args = parser.parse_args()

	csv_path = os.path.join("archive", "spam_ham_dataset.csv")
	train_csv = "train_data.csv"
	test_csv = "test_data.csv"
	vectorizer_path = "vectorizer.pkl"
	model_path = "model.pkl"
	metadata_path = "model_metadata.json"

	start_pipeline = time.time()

	try:
		download_nltk_resources()
	except Exception as e:
		logging.warning("NLTK download step encountered an issue: %s", e)

	# Phase 1: Preprocessing
	if args.phase in ("all", "preprocess"):
		need_preprocess = args.rerun_preprocess or not (validate_preprocessed_data(train_csv) and validate_preprocessed_data(test_csv))
		if not need_preprocess:
			logging.info("Preprocessed CSVs already exist and are valid. Skipping preprocessing.")
		else:
			try:
				df = load_dataset(csv_path)
			except Exception as e:
				logging.error("Failed to load dataset: %s", e)
				return

			try:
				perform_eda(df)
			except Exception as e:
				logging.warning("EDA step failed: %s", e)

			try:
				df = apply_preprocessing(df)
			except Exception as e:
				logging.error("Preprocessing failed: %s", e)
				return

			try:
				split_and_save(df, out_train=train_csv, out_test=test_csv)
			except Exception as e:
				logging.error("Failed to split/save data: %s", e)
				return

			# Validate outputs
			if not (validate_preprocessed_data(train_csv) and validate_preprocessed_data(test_csv)):
				logging.error("Preprocessed outputs failed validation after save.")
				return

			logging.info("Preprocessing pipeline completed successfully.")

	# Phase 2: TF-IDF
	if args.phase in ("all", "tfidf", "train", "select"):
		try:
			X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = extract_tfidf_features(
				train_csv_path=train_csv, test_csv_path=test_csv, vectorizer_output_path=vectorizer_path
			)
		except Exception as e:
			logging.error("TF-IDF extraction failed: %s", e)
			return

	# Phase 3: Training
	if args.phase in ("all", "train", "select"):
		try:
			results_df, fitted_models = train_and_compare_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
		except Exception as e:
			logging.error("Training phase failed: %s", e)
			return

	# Phase 4: Select and persist best model
	if args.phase in ("all", "select"):
		try:
			best_model, best_name, metadata = select_and_save_best_model(results_df, fitted_models, model_output_path=model_path, metadata_path=metadata_path, vectorizer=vectorizer)
			logging.info("Best model: %s", best_name)

			# Synchronize to web app if requested or auto-detected
			sync_dir = args.output_dir
			if not sync_dir:
				default_sync_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine", "spam-spotter-engine", "frontend", "public", "models"))
				if os.path.isdir(default_sync_path):
					sync_dir = default_sync_path
					logging.info("Auto-detected web app model directory: %s", sync_dir)

			if sync_dir:
				import shutil
				os.makedirs(sync_dir, exist_ok=True)
				for file_name in ("vectorizer.pkl", "model.pkl", "model_metadata.json"):
					src = file_name
					dst = os.path.join(sync_dir, file_name)
					logging.info("Syncing %s to %s", src, dst)
					shutil.copy2(src, dst)
				logging.info("Successfully synchronized model artifacts to %s", sync_dir)
		except Exception as e:
			logging.error("Model selection/persistence failed: %s", e)
			return

	total_time = time.time() - start_pipeline
	logging.info("Pipeline completed in %.2f seconds", total_time)


if __name__ == "__main__":
	main()

