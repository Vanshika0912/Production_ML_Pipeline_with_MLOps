"""
Model Evaluation Component
Comprehensive evaluation, reporting, and test-set assessment with MLflow logging.
"""

import os
import logging
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, log_loss, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    reports_dir: str = os.path.join("reports", "evaluation")
    metrics_path: str = os.path.join("reports", "evaluation", "metrics.json")
    confusion_matrix_path: str = os.path.join("reports", "evaluation", "confusion_matrix.png")
    roc_curve_path: str = os.path.join("reports", "evaluation", "roc_curve.png")
    classification_report_path: str = os.path.join("reports", "evaluation", "classification_report.txt")
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "production_ml_pipeline"
    promotion_threshold: float = 0.80       # Min F1 score to promote model to production
    target_column: str = "target"


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations and MLflow integration.
    Generates confusion matrices, ROC curves, precision-recall curves, and full reports.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_metrics: Dict[str, float] = {}
        Path(config.reports_dir).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        run_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Full evaluation of a model on the test set.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Core metrics
        metrics = self._compute_core_metrics(y_test, y_pred, y_prob)
        self.evaluation_metrics = metrics

        # Classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        self._save_classification_report(report, model_name)

        # Visualizations
        classes = sorted(np.unique(y_test))
        self._plot_confusion_matrix(y_test, y_pred, classes, model_name)

        if y_prob is not None and len(classes) == 2:
            self._plot_roc_curve(y_test, y_prob[:, 1], model_name)
            self._plot_precision_recall_curve(y_test, y_prob[:, 1], model_name)
            self._plot_calibration_curve(y_test, y_prob[:, 1], model_name)

        # Save metrics to JSON
        self._save_metrics(metrics, model_name)

        # Log to MLflow if run_id is provided
        if run_id:
            self._log_to_mlflow(metrics, model_name, run_id)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION SUMMARY — {model_name}")
        logger.info(f"{'='*60}")
        for k, v in metrics.items():
            logger.info(f"  {k:<25}: {v:.4f}")
        logger.info(f"{'='*60}")

        # Promotion check
        promote = metrics.get("f1", 0) >= self.config.promotion_threshold
        status = "✅ PROMOTE" if promote else "❌ REJECT"
        logger.info(f"  Promotion status ({self.config.promotion_threshold} threshold): {status}")

        return metrics

    def _compute_core_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred)
        }

        # Add a top-level f1 for promotion check
        metrics["f1"] = metrics["f1_weighted"]

        # Binary-specific metrics
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            metrics["avg_precision"] = average_precision_score(y_true, y_prob[:, 1])
            try:
                metrics["log_loss"] = log_loss(y_true, y_prob)
            except Exception:
                pass

        return metrics

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        classes: List, model_name: str
    ):
        """Generate and save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=classes, yticklabels=classes)
        axes[0].set_title(f"Confusion Matrix — {model_name}")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                    xticklabels=classes, yticklabels=classes)
        axes[1].set_title(f"Normalized Confusion Matrix — {model_name}")
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")

        plt.tight_layout()
        path = self.config.confusion_matrix_path.replace(".png", f"_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved: {path}")

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
        plt.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        path = self.config.roc_curve_path.replace(".png", f"_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved: {path}")

    def _plot_precision_recall_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str
    ):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        baseline = y_true.mean()

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR (AP = {ap:.3f})")
        plt.axhline(y=baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        path = os.path.join(self.config.reports_dir, f"pr_curve_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_calibration_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str
    ):
        """Plot probability calibration curve."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )

            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", color="steelblue",
                     label=f"{model_name}")
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Calibration Curve — {model_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            path = os.path.join(self.config.reports_dir, f"calibration_{model_name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate calibration curve: {e}")

    def _save_classification_report(self, report: str, model_name: str):
        """Save classification report to file."""
        path = self.config.classification_report_path.replace(
            ".txt", f"_{model_name}.txt"
        )
        with open(path, "w") as f:
            f.write(f"Classification Report — {model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        logger.info(f"Classification report saved: {path}")

    def _save_metrics(self, metrics: Dict[str, float], model_name: str):
        """Save evaluation metrics to JSON."""
        path = self.config.metrics_path.replace(".json", f"_{model_name}.json")
        with open(path, "w") as f:
            json.dump({"model": model_name, "metrics": metrics}, f, indent=2)
        logger.info(f"Metrics saved: {path}")

    def _log_to_mlflow(self, metrics: Dict[str, float], model_name: str, run_id: str):
        """Log evaluation metrics and artifacts to an existing MLflow run."""
        try:
            with mlflow.start_run(run_id=run_id):
                for k, v in metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.set_tag("evaluation_complete", "true")
                # Log plots as artifacts
                for path in Path(self.config.reports_dir).glob(f"*{model_name}*"):
                    mlflow.log_artifact(str(path))
                logger.info(f"Metrics logged to MLflow run: {run_id}")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")

    def compare_models(self, models_metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Compare multiple model evaluation results.
        Returns the name of the best model.
        """
        logger.info("\nModel Comparison on Test Set:")
        comparison_df = pd.DataFrame(models_metrics).T
        logger.info(f"\n{comparison_df.to_string()}")

        best_model = max(models_metrics, key=lambda m: models_metrics[m].get("f1", 0))
        logger.info(f"\n🏆 Best model: {best_model}")
        return best_model

    def check_model_promotion(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if model meets the promotion criteria.
        
        Returns:
            (should_promote: bool, reason: str)
        """
        f1 = metrics.get("f1", 0)
        accuracy = metrics.get("accuracy", 0)
        threshold = self.config.promotion_threshold

        if f1 >= threshold:
            reason = f"F1={f1:.4f} ≥ threshold={threshold}"
            logger.info(f"✅ Model meets promotion criteria: {reason}")
            return True, reason
        else:
            reason = f"F1={f1:.4f} < threshold={threshold}"
            logger.warning(f"❌ Model does not meet promotion criteria: {reason}")
            return False, reason
