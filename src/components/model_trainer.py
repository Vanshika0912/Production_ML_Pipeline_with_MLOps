"""
Model Trainer Component
Trains multiple ML models with hyperparameter tuning and MLflow experiment tracking.
"""

import os
import logging
import pickle
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    model_save_path: str = os.path.join("models", "artifacts", "model.pkl")
    experiment_name: str = "production_ml_pipeline"
    run_name_prefix: str = "training_run"
    mlflow_tracking_uri: str = "http://localhost:5000"
    cv_folds: int = 5
    scoring_metric: str = "f1"
    hyperparameter_search: str = "random"       # grid, random, none
    n_iter_search: int = 20
    random_state: int = 42
    n_jobs: int = -1
    models_to_train: List[str] = field(default_factory=lambda: [
        "random_forest", "gradient_boosting", "logistic_regression", "extra_trees"
    ])


class ModelTrainer:
    """
    Multi-model trainer with MLflow integration.
    Trains, evaluates, and compares multiple classifiers. Logs all experiments to MLflow.
    """

    # Model registry with default hyperparameter grids
    MODEL_REGISTRY: Dict[str, Any] = {
        "random_forest": {
            "model": RandomForestClassifier,
            "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"]
            }
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier,
            "default_params": {"n_estimators": 100, "random_state": 42},
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 1.0]
            }
        },
        "logistic_regression": {
            "model": LogisticRegression,
            "default_params": {"random_state": 42, "max_iter": 1000, "n_jobs": -1},
            "param_grid": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            }
        },
        "extra_trees": {
            "model": ExtraTreesClassifier,
            "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "svm": {
            "model": SVC,
            "default_params": {"probability": True, "random_state": 42},
            "param_grid": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        },
        "ada_boost": {
            "model": AdaBoostClassifier,
            "default_params": {"random_state": 42},
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5, 1.0]
            }
        },
        "knn": {
            "model": KNeighborsClassifier,
            "default_params": {"n_jobs": -1},
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "decision_tree": {
            "model": DecisionTreeClassifier,
            "default_params": {"random_state": 42},
            "param_grid": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"]
            }
        }
    }

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.training_results: Dict[str, Dict] = {}
        Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow tracking server."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLflow tracking URI: {self.config.mlflow_tracking_uri}")
            logger.info(f"MLflow experiment: {self.config.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup warning (using local): {e}")
            mlflow.set_tracking_uri("./mlruns")
            mlflow.set_experiment(self.config.experiment_name)

    def _compute_metrics(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        }

        if y_prob is not None and len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, y_prob)

        return metrics

    def _run_hyperparameter_search(
        self, model_class, param_grid: Dict, X_train: np.ndarray, y_train: np.ndarray,
        default_params: Dict
    ):
        """Run hyperparameter search (grid or random)."""
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        base_model = model_class(**default_params)

        if self.config.hyperparameter_search == "grid":
            logger.info("Running GridSearchCV...")
            searcher = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs, verbose=0
            )
        elif self.config.hyperparameter_search == "random":
            logger.info(f"Running RandomizedSearchCV (n_iter={self.config.n_iter_search})...")
            searcher = RandomizedSearchCV(
                base_model, param_grid, cv=cv,
                scoring=self.config.scoring_metric,
                n_iter=self.config.n_iter_search,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=0
            )
        else:
            # No search, use defaults
            base_model.fit(X_train, y_train)
            return base_model, default_params, None

        searcher.fit(X_train, y_train)
        logger.info(f"Best params: {searcher.best_params_}")
        logger.info(f"Best CV score: {searcher.best_score_:.4f}")
        return searcher.best_estimator_, searcher.best_params_, searcher.best_score_

    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train a single model and log to MLflow."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_REGISTRY.keys())}")

        model_info = self.MODEL_REGISTRY[model_name]
        logger.info(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}")

        run_name = f"{self.config.run_name_prefix}_{model_name}"

        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.time()

            # Log training parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv_folds", self.config.cv_folds)
            mlflow.log_param("hp_search", self.config.hyperparameter_search)
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            # Hyperparameter search or default training
            model, best_params, cv_score = self._run_hyperparameter_search(
                model_info["model"],
                model_info["param_grid"],
                X_train, y_train,
                model_info["default_params"]
            )

            # Log best params
            for k, v in best_params.items():
                mlflow.log_param(f"best_{k}", v)

            # Cross-validation score
            if cv_score is None:
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.config.scoring_metric)
                cv_score = cv_scores.mean()
                mlflow.log_metric("cv_score_std", cv_scores.std())

            mlflow.log_metric("cv_score_mean", cv_score)

            # Evaluate on validation set
            val_metrics = self._compute_metrics(model, X_val, y_val)
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)

            # Training metrics
            train_metrics = self._compute_metrics(model, X_train, y_train)
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)

            # Training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)

            # Log the model to MLflow
            mlflow.sklearn.log_model(
                model, 
                artifact_path=f"models/{model_name}",
                registered_model_name=f"{self.config.experiment_name}_{model_name}"
            )

            # Log tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("pipeline_stage", "training")

            result = {
                "model": model,
                "model_name": model_name,
                "best_params": best_params,
                "cv_score": cv_score,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "training_time": training_time,
                "run_id": run.info.run_id
            }

            logger.info(f"✅ {model_name}: Val F1={val_metrics.get('f1', 0):.4f}, "
                       f"Val Acc={val_metrics.get('accuracy', 0):.4f}, "
                       f"Time={training_time:.2f}s")
            return result

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Dict]:
        """Train all configured models and select the best one."""
        logger.info(f"Training {len(self.config.models_to_train)} models...")
        logger.info(f"Models: {self.config.models_to_train}")

        results = {}
        for model_name in self.config.models_to_train:
            try:
                result = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
                results[model_name] = result

                # Track best model
                primary_metric = result["val_metrics"].get(self.config.scoring_metric,
                                                            result["val_metrics"].get("f1", 0))
                if primary_metric > self.best_score:
                    self.best_score = primary_metric
                    self.best_model = result["model"]
                    self.best_model_name = model_name

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}", exc_info=True)

        self.training_results = results

        # Print comparison table
        self._print_comparison_table(results)

        logger.info(f"\n🏆 Best model: {self.best_model_name} "
                   f"(Val {self.config.scoring_metric}: {self.best_score:.4f})")
        return results

    def _print_comparison_table(self, results: Dict[str, Dict]):
        """Print a comparison table of all trained models."""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*80)
        header = f"{'Model':<25} {'Val Acc':>10} {'Val F1':>10} {'Val AUC':>10} {'Train F1':>10} {'Time(s)':>10}"
        logger.info(header)
        logger.info("-"*80)

        for name, result in sorted(
            results.items(),
            key=lambda x: x[1]["val_metrics"].get("f1", 0),
            reverse=True
        ):
            vm = result["val_metrics"]
            tm = result["train_metrics"]
            row = (f"{name:<25} "
                   f"{vm.get('accuracy', 0):>10.4f} "
                   f"{vm.get('f1', 0):>10.4f} "
                   f"{vm.get('roc_auc', 0):>10.4f} "
                   f"{tm.get('f1', 0):>10.4f} "
                   f"{result['training_time']:>10.2f}")
            logger.info(row)
        logger.info("="*80)

    def save_model(self, model=None, filepath: Optional[str] = None):
        """Save the model to disk."""
        model_to_save = model or self.best_model
        path = filepath or self.config.model_save_path

        if model_to_save is None:
            raise RuntimeError("No model to save. Train models first.")

        artifacts = {
            "model": model_to_save,
            "model_name": self.best_model_name,
            "best_score": self.best_score,
            "training_results_summary": {
                name: {
                    "val_metrics": r["val_metrics"],
                    "cv_score": r["cv_score"],
                    "training_time": r["training_time"],
                    "run_id": r["run_id"]
                }
                for name, r in self.training_results.items()
            }
        }

        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info(f"✅ Best model '{self.best_model_name}' saved to: {path}")

    def load_model(self, filepath: Optional[str] = None):
        """Load a model from disk."""
        path = filepath or self.config.model_save_path
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        self.best_model = artifacts["model"]
        self.best_model_name = artifacts["model_name"]
        self.best_score = artifacts["best_score"]
        logger.info(f"Model '{self.best_model_name}' loaded from: {path}")
        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the best trained model."""
        if self.best_model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.best_model is None:
            raise RuntimeError("No model loaded.")
        if not hasattr(self.best_model, "predict_proba"):
            raise RuntimeError(f"Model {self.best_model_name} does not support predict_proba")
        return self.best_model.predict_proba(X)
