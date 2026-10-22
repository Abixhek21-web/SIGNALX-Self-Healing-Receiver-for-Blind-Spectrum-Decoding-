"""
Training Execution Script.
Executes dataset ingestion, multi-view feature extraction, and ML/CNN model training.
"""

from pathlib import Path
import sys
import os

# Add workspace to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.train_model import train_models
from training.synthetic_generator import generate_ground_truth_dataset
from utils.logging import logger


def main():
    logger.info("==================================================")
    logger.info("SIH 26147 — AUTOMATED RF MODEL TRAINING PIPELINE")
    logger.info("==================================================")

    train_path = r"C:\Users\Abishek\Downloads\rfc_dataset\dataset\demod_train"
    val_path = r"C:\Users\Abishek\Downloads\rfc_dataset\dataset\demod_val"

    # Step 1: Generate synthetic ground truth benchmark dataset
    logger.info("Generating synthetic ground truth benchmark records...")
    generate_ground_truth_dataset(
        output_json_path="reports/ground_truth.json",
        num_samples_per_mod=25,
        seed=42,
    )

    # Step 2: Train models
    logger.info("Starting model training on demod_train and synthetic datasets...")
    training_report = train_models(
        train_dir=train_path if os.path.exists(train_path) else None,
        val_dir=val_path if os.path.exists(val_path) else None,
        models_dir="models",
        reports_dir="reports",
        random_seed=42,
        synthetic_samples_per_mod_train=120,
        synthetic_samples_per_mod_val=30,
        rf_train_limit=60,
        rf_val_limit=20,
    )

    logger.info("==================================================")
    logger.info("TRAINING PIPELINE SUCCESSFULLY COMPLETED")
    logger.info(f"Random Forest Validation Acc: {training_report['random_forest']['validation_accuracy']:.2%}")
    logger.info(f"1D-CNN Validation Acc:        {training_report['neural_network_1d_cnn']['validation_accuracy']:.2%}")
    logger.info("Saved Models: models/modulation_rf.joblib, models/modulation_cnn.pt")
    logger.info("Saved Report: reports/training_report.json")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
