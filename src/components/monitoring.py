"""
Monitoring and Logging Component
Tracks model performance and detects data/concept drift.
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""
    log_file: str = "logs/model_monitoring.log"
    drift_threshold: float = 0.1  # KS test statistic threshold
    alert_email: str = "alerts@company.com"


class ModelMonitor:
    """
    Monitors model performance and detects data drift.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            filename=self.config.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger.info("Monitoring system initialized")

    def log_inference(self, features: pd.DataFrame, prediction: float, model_version: str):
        """Log inference events."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "prediction": prediction,
            "features_summary": features.describe().to_dict()
        }
        logger.info(f"Inference logged: {log_entry}")

    def detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Simple data drift detection using Kolmogorov-Smirnov test.
        """
        from scipy.stats import ks_2samp
        
        drift_report = {}
        for col in reference_data.columns:
            if col in current_data.columns and reference_data[col].dtype in ['float64', 'int64']:
                stat, p_value = ks_2samp(reference_data[col], current_data[col])
                drift_detected = stat > self.config.drift_threshold
                drift_report[col] = drift_detected
                if drift_detected:
                    logger.warning(f"Drift detected in feature '{col}' (stat={stat:.4f})")
        
        return drift_report
