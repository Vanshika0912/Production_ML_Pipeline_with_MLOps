"""
Data Ingestion Component
Handles loading data from various sources: CSV, databases, APIs, S3
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    raw_data_path: str = os.path.join("data", "raw", "dataset.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")
    val_data_path: str = os.path.join("data", "processed", "val.csv")
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    target_column: str = "target"


class DataIngestion:
    """
    Data Ingestion component for loading, validating, and splitting datasets.
    Supports CSV files, databases (via SQLAlchemy), and cloud storage (S3).
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.config.train_data_path, self.config.test_data_path, self.config.val_data_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.raw_data_path).parent.mkdir(parents=True, exist_ok=True)

    def load_from_csv(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load data from a CSV file."""
        path = filepath or self.config.raw_data_path
        logger.info(f"Loading data from CSV: {path}")
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise

    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from a SQL database using SQLAlchemy."""
        try:
            from sqlalchemy import create_engine
            logger.info("Loading data from database...")
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except ImportError:
            logger.error("SQLAlchemy not installed. Run: pip install sqlalchemy")
            raise

    def load_from_s3(self, bucket: str, key: str, aws_credentials: Optional[dict] = None) -> pd.DataFrame:
        """Load data from AWS S3."""
        try:
            import boto3
            logger.info(f"Loading data from S3: s3://{bucket}/{key}")
            s3_client = boto3.client("s3", **(aws_credentials or {}))
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(obj["Body"])
            logger.info(f"Loaded {len(df)} rows from S3")
            return df
        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
            raise

    def generate_synthetic_data(self, n_samples: int = 5000, n_features: int = 20) -> pd.DataFrame:
        """
        Generate synthetic classification dataset for demonstration.
        Used when no real data is available.
        """
        logger.info(f"Generating synthetic dataset: {n_samples} samples, {n_features} features")
        np.random.seed(self.config.random_state)

        # Create feature columns
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X = np.random.randn(n_samples, n_features)

        # Create target with some pattern (binary classification)
        weights = np.random.randn(n_features)
        logits = X @ weights + np.random.randn(n_samples) * 0.5
        y = (logits > 0).astype(int)

        df = pd.DataFrame(X, columns=feature_names)
        df[self.config.target_column] = y

        # Add some noise columns and missing values to simulate real data
        df["categorical_col"] = np.random.choice(["A", "B", "C", "D"], size=n_samples)
        df.loc[np.random.choice(df.index, size=int(0.05 * n_samples), replace=False), "feature_0"] = np.nan

        logger.info(f"Generated dataset with shape: {df.shape}")
        logger.info(f"Class distribution:\n{df[self.config.target_column].value_counts()}")
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data for quality checks.
        Returns True if data is valid, raises ValueError otherwise.
        """
        logger.info("Running data validation checks...")
        issues = []

        # Check for empty dataframe
        if df.empty:
            issues.append("DataFrame is empty")

        # Check for target column
        if self.config.target_column not in df.columns:
            issues.append(f"Target column '{self.config.target_column}' not found")

        # Check for excessive missing values (>50% in any column)
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing:
            issues.append(f"Columns with >50% missing values: {high_missing}")

        # Check for sufficient data
        if len(df) < 100:
            issues.append(f"Insufficient data: only {len(df)} rows")

        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate rows")

        if issues:
            for issue in issues:
                logger.error(f"Validation failed: {issue}")
            raise ValueError(f"Data validation failed: {'; '.join(issues)}")

        logger.info("✅ Data validation passed")
        return True

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        Returns (train_df, val_df, test_df).
        """
        logger.info(f"Splitting data: test={self.config.test_size}, val={self.config.val_size}")

        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[self.config.target_column] if df[self.config.target_column].nunique() <= 20 else None
        )

        # Second split: train vs val
        adjusted_val_size = self.config.val_size / (1 - self.config.test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val_size,
            random_state=self.config.random_state,
            stratify=train_val[self.config.target_column] if train_val[self.config.target_column].nunique() <= 20 else None
        )

        logger.info(f"Split sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Save data splits to disk."""
        train.to_csv(self.config.train_data_path, index=False)
        val.to_csv(self.config.val_data_path, index=False)
        test.to_csv(self.config.test_data_path, index=False)
        logger.info(f"Saved train to: {self.config.train_data_path}")
        logger.info(f"Saved val to: {self.config.val_data_path}")
        logger.info(f"Saved test to: {self.config.test_data_path}")

    def run(self, source: str = "synthetic", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the full data ingestion pipeline.
        
        Args:
            source: 'csv', 'database', 's3', or 'synthetic'
            **kwargs: Additional arguments for the specific loader
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Starting data ingestion from source: {source}")

        if source == "csv":
            df = self.load_from_csv(kwargs.get("filepath"))
        elif source == "database":
            df = self.load_from_database(kwargs["connection_string"], kwargs["query"])
        elif source == "s3":
            df = self.load_from_s3(kwargs["bucket"], kwargs["key"])
        elif source == "synthetic":
            df = self.generate_synthetic_data(
                n_samples=kwargs.get("n_samples", 5000),
                n_features=kwargs.get("n_features", 20)
            )
            # Save synthetic data as raw
            df.to_csv(self.config.raw_data_path, index=False)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'csv', 'database', 's3', or 'synthetic'")

        # Validate
        self.validate_data(df)

        # Split
        train, val, test = self.split_data(df)

        # Save
        self.save_splits(train, val, test)

        logger.info("✅ Data ingestion completed successfully")
        return train, val, test
