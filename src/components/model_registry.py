"""
Model Registry Component
Manages model versioning, staging, and promotion using MLflow Model Registry.
"""

import logging
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry operations."""
    mlflow_tracking_uri: str = "http://localhost:5000"
    registry_model_name: str = "production_ml_pipeline_model"
    staging_stage: str = "Staging"
    production_stage: str = "Production"
    archive_stage: str = "Archived"
    min_f1_for_staging: float = 0.75
    min_f1_for_production: float = 0.80


class ModelRegistry:
    """
    MLflow Model Registry integration.
    Handles registering, versioning, promoting, and transitioning models
    through Staging → Production → Archived lifecycle stages.
    """

    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        try:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        except Exception:
            mlflow.set_tracking_uri("./mlruns")
        self.client = MlflowClient()

    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a model from an MLflow run into the Model Registry.
        
        Args:
            run_id: MLflow run ID where the model was logged
            artifact_path: Path to the model artifact within the run
            model_name: Registry model name (uses config default if None)
            tags: Optional metadata tags
            description: Optional model version description
        
        Returns:
            ModelVersion object
        """
        name = model_name or self.config.registry_model_name
        model_uri = f"runs:/{run_id}/{artifact_path}"

        logger.info(f"Registering model '{name}' from run: {run_id}")
        logger.info(f"Model URI: {model_uri}")

        try:
            result = mlflow.register_model(model_uri=model_uri, name=name)
            version = result.version

            logger.info(f"✅ Model registered: {name} v{version}")

            # Add tags
            if tags:
                for k, v in tags.items():
                    self.client.set_model_version_tag(name, version, k, v)

            # Add description
            if description:
                self.client.update_model_version(
                    name=name,
                    version=version,
                    description=description
                )

            # Add timestamp tag
            self.client.set_model_version_tag(
                name, version, "registered_at",
                datetime.utcnow().isoformat()
            )

            return result

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> ModelVersion:
        """
        Transition a model version to a new lifecycle stage.
        
        Stages: None → Staging → Production → Archived
        """
        logger.info(f"Transitioning {model_name} v{version} → {stage}")

        result = self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )

        logger.info(f"✅ Model {model_name} v{version} is now in {stage}")
        return result

    def promote_to_staging(
        self, model_name: str, version: str, metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Promote a model version to Staging if it meets the threshold."""
        f1 = metrics.get("f1", 0)
        threshold = self.config.min_f1_for_staging

        if f1 < threshold:
            reason = f"F1={f1:.4f} < staging threshold={threshold}"
            logger.warning(f"❌ Cannot promote to Staging: {reason}")
            return False, reason

        self.transition_model_stage(model_name, version, self.config.staging_stage)
        reason = f"F1={f1:.4f} ≥ staging threshold={threshold}"
        logger.info(f"✅ Promoted to Staging: {reason}")
        return True, reason

    def promote_to_production(
        self, model_name: str, version: str, metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Promote a model version to Production if it meets the threshold."""
        f1 = metrics.get("f1", 0)
        threshold = self.config.min_f1_for_production

        if f1 < threshold:
            reason = f"F1={f1:.4f} < production threshold={threshold}"
            logger.warning(f"❌ Cannot promote to Production: {reason}")
            return False, reason

        self.transition_model_stage(
            model_name, version, self.config.production_stage,
            archive_existing=True
        )
        reason = f"F1={f1:.4f} ≥ production threshold={threshold}"
        logger.info(f"🚀 Promoted to Production: {reason}")
        return True, reason

    def get_production_model(self, model_name: Optional[str] = None):
        """Load and return the current production model."""
        name = model_name or self.config.registry_model_name

        try:
            versions = self.client.get_latest_versions(
                name, stages=[self.config.production_stage]
            )
            if not versions:
                logger.warning(f"No production model found for: {name}")
                return None, None

            prod_version = versions[0]
            model_uri = f"models:/{name}/{self.config.production_stage}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Loaded production model: {name} v{prod_version.version}")
            return model, prod_version

        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None, None

    def get_staging_model(self, model_name: Optional[str] = None):
        """Load and return the current staging model."""
        name = model_name or self.config.registry_model_name

        try:
            versions = self.client.get_latest_versions(
                name, stages=[self.config.staging_stage]
            )
            if not versions:
                logger.warning(f"No staging model found for: {name}")
                return None, None

            staging_version = versions[0]
            model_uri = f"models:/{name}/{self.config.staging_stage}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Loaded staging model: {name} v{staging_version.version}")
            return model, staging_version

        except Exception as e:
            logger.error(f"Failed to load staging model: {e}")
            return None, None

    def list_model_versions(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """List all versions of a registered model."""
        name = model_name or self.config.registry_model_name
        try:
            versions = self.client.search_model_versions(f"name='{name}'")
            logger.info(f"\nModel versions for '{name}':")
            for v in versions:
                logger.info(f"  v{v.version}: {v.current_stage} | Run: {v.run_id[:8]}... | {v.status}")
            return list(versions)
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []

    def get_model_version_metrics(self, model_name: str, version: str) -> Dict:
        """Get metrics logged for a specific model version."""
        try:
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)
            return {
                "version": version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
        except Exception as e:
            logger.error(f"Failed to get metrics for v{version}: {e}")
            return {}

    def compare_staging_vs_production(
        self, model_name: Optional[str] = None
    ) -> Dict:
        """Compare staging model metrics against production model."""
        name = model_name or self.config.registry_model_name

        staging_versions = self.client.get_latest_versions(name, stages=["Staging"])
        prod_versions = self.client.get_latest_versions(name, stages=["Production"])

        comparison = {}
        if staging_versions:
            comparison["staging"] = self.get_model_version_metrics(
                name, staging_versions[0].version
            )
        if prod_versions:
            comparison["production"] = self.get_model_version_metrics(
                name, prod_versions[0].version
            )

        if comparison.get("staging") and comparison.get("production"):
            staging_f1 = comparison["staging"]["metrics"].get("val_f1", 0)
            prod_f1 = comparison["production"]["metrics"].get("val_f1", 0)
            comparison["recommendation"] = (
                "PROMOTE_STAGING"
                if staging_f1 > prod_f1
                else "KEEP_PRODUCTION"
            )
            logger.info(
                f"Staging F1={staging_f1:.4f} vs Production F1={prod_f1:.4f} → "
                f"{comparison['recommendation']}"
            )

        return comparison

    def archive_model_version(self, model_name: str, version: str):
        """Archive a specific model version."""
        self.transition_model_stage(
            model_name, version, self.config.archive_stage, archive_existing=False
        )
        logger.info(f"Archived model {model_name} v{version}")
