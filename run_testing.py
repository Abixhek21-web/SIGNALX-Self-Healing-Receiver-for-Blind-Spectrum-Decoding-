"""
Testing and Evaluation Execution Script.
Executes testing across all 15 stages: DSP baseline, ML model, Hybrid fusion,
SNR sweep, parameter estimation, demodulation BER, FEC codecs, interleavers,
adaptive search, and RFChallenge validation.
"""

from pathlib import Path
import sys
import os
import numpy as np

# Add workspace to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.evaluate_model import ModelEvaluator
from utils.logging import logger


def main():
    logger.info("==================================================")
    logger.info("SIH 26147 — AUTOMATED MODEL TESTING & EVALUATION")
    logger.info("==================================================")

    evaluator = ModelEvaluator(models_dir="models", reports_dir="reports", random_seed=42)

    # 1. Load validation feature sets
    val_feat_path = Path("data/features/val_features.npy")
    val_lbl_path = Path("data/features/val_labels.npy")

    if val_feat_path.exists() and val_lbl_path.exists():
        X_val = np.load(val_feat_path)
        y_val = np.load(val_lbl_path)
        logger.info(f"Loaded validation set: {len(X_val)} samples.")
    else:
        logger.warning("Validation features not found, extracting temporary validation benchmark set...")
        from training.feature_extractor import FeatureExtractorPipeline
        ext = FeatureExtractorPipeline()
        X_val, y_val = ext.build_dataset_features(
            rf_dataset=None, synthetic_samples_per_mod=30, seed=1234
        )

    # 2. Evaluate Classification & Generate Confusion Matrix
    logger.info("Running Classification Evaluation (DSP vs ML vs Hybrid)...")
    cls_res = evaluator.evaluate_classification(X_val, y_val, dsp_weight=0.50, ml_weight=0.50)

    # 3. SNR Sweep Analysis
    logger.info("Running SNR Sweep Evaluation (-10 dB to +25 dB)...")
    snr_res = evaluator.evaluate_snr_sweep()

    # 4. Parameter Estimation Testing
    logger.info("Testing Parameter Estimation (Symbol Rate, CFO, Timing)...")
    param_res = evaluator.evaluate_parameter_estimation(num_trials=30)

    # 5. Demodulation Testing (BER / SER)
    logger.info("Testing Demodulation Engine across BPSK, QPSK, 8PSK, 16QAM, 2FSK...")
    demod_res = evaluator.evaluate_demodulation()

    # 6. FEC Codec Testing
    logger.info("Testing FEC Decoders (Viterbi, Reed-Solomon, Concatenated, LDPC)...")
    fec_res = evaluator.evaluate_fec()

    # 7. Interleaver Testing
    logger.info("Testing Interleaver Bank (Block, Conv, Diagonal, PR)...")
    il_res = evaluator.evaluate_interleavers()

    # 8. Adaptive Search Testing
    logger.info("Testing Adaptive Search Multi-Hypothesis Retry Loop...")
    search_res = evaluator.evaluate_adaptive_search()

    # 9. Real RFChallenge Validation Test
    logger.info("Testing RFChallenge Real Data Validation on demod_val...")
    rfc_val_path = r"C:\Users\Abishek\Downloads\rfc_dataset\dataset\demod_val"
    rfc_res = evaluator.evaluate_rfchallenge_real_data(val_dir=rfc_val_path, num_samples=15)

    logger.info("==================================================")
    logger.info("TESTING COMPLETED SUCCESSFULLY")
    logger.info(f"Hybrid Accuracy:    {cls_res['hybrid_accuracy']:.2%}")
    logger.info(f"Top-3 Accuracy:     {cls_res['top3_accuracy']:.2%}")
    logger.info(f"Symbol Rate RMSE:   {param_res['symbol_rate']['rmse_hz']} Hz")
    logger.info(f"CFO RMSE:           {param_res['cfo_estimation']['rmse_hz']} Hz")
    logger.info(f"RFChallenge Passed: {rfc_res.get('successful_analyses', 0)}/{rfc_res.get('validation_samples_tested', 0)}")
    logger.info("Generated Reports in reports/")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
