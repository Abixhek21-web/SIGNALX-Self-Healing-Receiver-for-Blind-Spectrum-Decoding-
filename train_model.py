"""
ML Modulation Recognition Training Script.
Generates synthetic dataset across SNRs (0 to 30 dB), CFOs, phase offsets, and timing offsets.
Extracts multi-view features and trains a Random Forest Classifier.
Saves model bundle to models/modulation_rf.joblib.
"""

from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from data_generator import generate_signal, SUPPORTED_MODS
from features import extract_all_features, FEATURE_NAMES
from utils.logging import logger


def train_modulation_model(
    samples_per_class: int = 40,
    save_path: str = "models/modulation_rf.joblib",
    random_seed: int = 42
):
    """
    Generate dataset, extract features, train Random Forest, evaluate and persist model.
    """
    logger.info(f"Starting ML model training ({samples_per_class} signals per class across {len(SUPPORTED_MODS)} modulations)...")
    np.random.seed(random_seed)

    X_list = []
    y_list = []

    start_time = time.time()
    total_signals = samples_per_class * len(SUPPORTED_MODS)
    count = 0

    snr_range = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    symbol_rates = [100000.0, 200000.0, 250000.0, 500000.0]
    sample_rate = 1e6

    for mod in SUPPORTED_MODS:
        for i in range(samples_per_class):
            snr = float(np.random.choice(snr_range))
            rs = float(np.random.choice(symbol_rates))
            cfo = float(np.random.uniform(-25000.0, 25000.0))
            phase = float(np.random.uniform(-np.pi, np.pi))
            timing = float(np.random.uniform(0.0, 2.0))
            num_syms = int(np.random.choice([400, 600, 800, 1000]))

            sig, _ = generate_signal(
                modulation=mod,
                num_symbols=num_syms,
                sample_rate=sample_rate,
                symbol_rate=rs,
                snr_db=snr,
                cfo_hz=cfo,
                phase_offset_rad=phase,
                timing_offset_samples=timing,
                seed=random_seed + count
            )

            # Feature extraction
            _, feat_vector, _ = extract_all_features(sig, sample_rate)

            # Replace NaNs
            feat_clean = np.nan_to_num(feat_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            X_list.append(feat_clean)
            y_list.append(mod)
            count += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    logger.info(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features.")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=3,
        random_state=random_seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    logger.info(f"Training Complete! Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{report}")

    # Persist model bundle
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_names": FEATURE_NAMES,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "trained_timestamp": time.time()
    }
    joblib.dump(bundle, save_file)
    logger.info(f"Model saved successfully to {save_path} in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    train_modulation_model()
