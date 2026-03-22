"""
Data Preprocessing Component
Handles feature engineering, encoding, scaling, and transformation pipelines.
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    preprocessor_path: str = os.path.join("models", "artifacts", "preprocessor.pkl")
    target_column: str = "target"
    numerical_imputer_strategy: str = "median"   # mean, median, most_frequent, constant
    categorical_imputer_strategy: str = "most_frequent"
    scaler_type: str = "standard"                # standard, minmax, robust
    encoding_type: str = "onehot"                # onehot, ordinal, label
    feature_selection: bool = True
    n_features_to_select: int = 15
    apply_pca: bool = False
    pca_components: int = 10
    handle_outliers: bool = True
    outlier_threshold: float = 3.0               # Z-score threshold


class DataPreprocessing:
    """
    Data Preprocessing component that builds sklearn transformation pipelines.
    Automatically detects numeric vs. categorical columns and applies appropriate transforms.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.pca: Optional[PCA] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names_out: List[str] = []
        Path(config.preprocessor_path).parent.mkdir(parents=True, exist_ok=True)

    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically detect numerical and categorical columns."""
        target = self.config.target_column
        feature_cols = [c for c in df.columns if c != target]

        numerical_cols = df[feature_cols].select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        categorical_cols = df[feature_cols].select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        logger.info(f"Detected {len(numerical_cols)} numerical columns")
        logger.info(f"Detected {len(categorical_cols)} categorical columns")
        return numerical_cols, categorical_cols

    def _get_scaler(self):
        """Return the configured scaler."""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        return scalers.get(self.config.scaler_type, StandardScaler())

    def _get_encoder(self):
        """Return the configured encoder."""
        if self.config.encoding_type == "onehot":
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif self.config.encoding_type == "ordinal":
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            return OrdinalEncoder()

    def handle_outliers(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Cap outliers using Z-score method."""
        logger.info(f"Handling outliers with Z-score threshold: {self.config.outlier_threshold}")
        df_clean = df.copy()
        for col in numerical_cols:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    lower = mean - self.config.outlier_threshold * std
                    upper = mean + self.config.outlier_threshold * std
                    n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
                    if n_outliers > 0:
                        logger.debug(f"Capping {n_outliers} outliers in column '{col}'")
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        return df_clean

    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build the sklearn preprocessing pipeline."""
        numerical_cols, categorical_cols = self._detect_column_types(df)

        # Numeric pipeline: impute → scale
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.config.numerical_imputer_strategy)),
            ("scaler", self._get_scaler())
        ])

        # Categorical pipeline: impute → encode
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.config.categorical_imputer_strategy)),
            ("encoder", self._get_encoder())
        ])

        transformers = []
        if numerical_cols:
            transformers.append(("num", numeric_pipeline, numerical_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        logger.info("✅ Preprocessing pipeline built")
        return self.preprocessor

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and transform it.
        Also fits feature selector and PCA if configured.
        
        Returns:
            (X_train_transformed, y_train)
        """
        logger.info("Fitting and transforming training data...")

        target = self.config.target_column
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target].values

        numerical_cols, _ = self._detect_column_types(train_df)

        # Handle outliers
        if self.config.handle_outliers:
            train_df = self.handle_outliers(train_df, numerical_cols)

        # Build and fit preprocessor
        self.build_preprocessor(train_df)
        X_transformed = self.preprocessor.fit_transform(X_train)

        logger.info(f"After preprocessing: shape = {X_transformed.shape}")

        # Feature selection
        if self.config.feature_selection:
            n_features = min(self.config.n_features_to_select, X_transformed.shape[1])
            logger.info(f"Applying feature selection: top {n_features} features")
            self.feature_selector = SelectKBest(
                score_func=f_classif if len(np.unique(y_train)) <= 20 else mutual_info_classif,
                k=n_features
            )
            X_transformed = self.feature_selector.fit_transform(X_transformed, y_train)
            logger.info(f"After feature selection: shape = {X_transformed.shape}")

        # PCA
        if self.config.apply_pca:
            n_components = min(self.config.pca_components, X_transformed.shape[1])
            logger.info(f"Applying PCA: {n_components} components")
            self.pca = PCA(n_components=n_components, random_state=42)
            X_transformed = self.pca.fit_transform(X_transformed)
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_var:.3f}")

        logger.info(f"Final training shape: {X_transformed.shape}")
        return X_transformed, y_train

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using the fitted preprocessor.
        
        Returns:
            (X_transformed, y) — y is None if target column is not present
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")

        target = self.config.target_column
        has_target = target in df.columns

        X = df.drop(columns=[target]) if has_target else df.copy()
        y = df[target].values if has_target else None

        # Apply preprocessor
        X_transformed = self.preprocessor.transform(X)

        # Apply feature selection
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)

        # Apply PCA
        if self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)

        return X_transformed, y

    def save_preprocessor(self):
        """Persist the fitted preprocessor to disk."""
        artifacts = {
            "preprocessor": self.preprocessor,
            "feature_selector": self.feature_selector,
            "pca": self.pca,
            "config": self.config
        }
        with open(self.config.preprocessor_path, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info(f"Preprocessor saved to: {self.config.preprocessor_path}")

    def load_preprocessor(self):
        """Load a previously fitted preprocessor from disk."""
        with open(self.config.preprocessor_path, "rb") as f:
            artifacts = pickle.load(f)
        self.preprocessor = artifacts["preprocessor"]
        self.feature_selector = artifacts["feature_selector"]
        self.pca = artifacts["pca"]
        self.config = artifacts["config"]
        logger.info(f"Preprocessor loaded from: {self.config.preprocessor_path}")

    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """Return feature importance scores from feature selector."""
        if self.feature_selector is None:
            return None
        scores = self.feature_selector.scores_
        pvalues = self.feature_selector.pvalues_
        support = self.feature_selector.get_support()
        return pd.DataFrame({
            "score": scores,
            "pvalue": pvalues,
            "selected": support
        }).sort_values("score", ascending=False)
