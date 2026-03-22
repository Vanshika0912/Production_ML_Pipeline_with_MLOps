# Production ML Pipeline with MLOps

A modular, scalable, and production-grade Machine Learning pipeline designed for the complete ML lifecycle: from raw data ingestion to automated model deployment and monitoring.

## 🚀 Key Features

*   **Robust Data Engineering**:
    *   Unified data ingestion for CSV, SQL Databases, and AWS S3.
    *   Automated data validation and stratified train/val/test splitting.
    *   Advanced preprocessing pipeline (imputation, scaling, encoding, outlier handling).
*   **Automated Model Training & Tuning**:
    *   Multi-model training (Random Forest, Gradient Boosting, Logistic Regression, etc.).
    *   Integrated hyperparameter optimization (GridSearch / RandomSearch).
    *   Seamless MLflow experiment tracking.
*   **Model Management (MLOps)**:
    *   MLflow Model Registry for version control and lifecycle management (Staging/Production/Archived).
    *   Automated model promotion based on performance thresholds.
*   **Production Deployment**:
    *   FastAPI-based REST API for high-performance model serving.
    *   Docker-ready with modular architecture for easy scaling.
*   **Monitoring & Observability**:
    *   Inference logging for performance tracking.
    *   Automated data drift detection using Kolmogorov-Smirnov (KS) tests.

## 🏗️ Architecture

```mermaid
graph LR
    Data[Data Source] --> Ingestion[Data Ingestion]
    Ingestion --> Preproc[Preprocessing]
    Preproc --> Training[Training & Tuning]
    Training --> Registry[MLflow Registry]
    Registry --> Serving[FastAPI Service]
    Serving --> Monitoring[Monitoring & Drift Detection]
```

## 🛠️ Project Structure

```text
Production_ML_Pipeline_with_MLOps/
├── app/                  # FastAPI serving application
├── src/components/       # Modular ML components
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   ├── model_registry.py
│   └── monitoring.py
├── models/artifacts/     # Serialized models and preprocessors
├── reports/evaluation/   # Model performance metrics & plots
├── Dockerfile            # Containerization
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.9+

### 2. Setup & Installation
```bash
# Clone the repository
git clone <repo-url>
cd Production_ML_Pipeline_with_MLOps

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Service
```bash
# Build the Docker image
docker build -t production-ml-pipeline .

# Run the container
docker run -p 8000:8000 production-ml-pipeline
```

The API will be available at `http://localhost:8000`. Access `http://localhost:8000/docs` for the interactive API documentation.

## 📈 ML Lifecycle Flow
1.  **Ingest**: Load and validate data.
2.  **Preprocess**: Fit transformation pipelines.
3.  **Train**: Run multiple experiments, log metrics to MLflow, and tune hyperparameters.
4.  **Evaluate**: Assess test-set performance, generate plots, and check promotion criteria.
5.  **Register/Deploy**: Promote the best model to Production.
6.  **Monitor**: Log live predictions and detect feature distribution drift.
