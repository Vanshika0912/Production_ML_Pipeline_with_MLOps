# ⚙️ Production ML Pipeline with MLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A modular, scalable, and production-grade Machine Learning pipeline designed for the complete ML lifecycle: from raw data ingestion to automated model deployment and monitoring.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Components](#-components) • [MLOps Lifecycle](#-mlops-lifecycle)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Components](#-components)
- [MLOps Lifecycle](#-mlops-lifecycle)
- [Deployment](#-deployment)

---

## 📖 Project Overview

This project provides a robust framework for building and maintaining production ML pipelines. It abstracts the complexity of data handling, model training, experiment tracking, and serving into reusable components, ensuring that models are not just trained but also tracked, validated, and deployed reliably.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🔍 **Unified Data Ingestion** | Support for CSV, SQL Databases, and AWS S3 with built-in validation & splitting. |
| 🛠️ **Automated Preprocessing** | Scikit-learn pipelines with automated scaling, imputation, encoding, & feature engineering. |
| 🧠 **Multi-Model Training** | Support for RF, GradientBoosting, LogisticRegression, ExtraTrees with hyperparameter tuning. |
| 📈 **MLflow Integration** | Full experiment tracking, model versioning, and lifecycle management (registry). |
| 🚀 **High-Performance Serving** | FastAPI-based REST API for scalable model inference. |
| 🐳 **Docker-Ready** | Optimized containerization for consistent environment deployment. |
| 📊 **Monitoring & Observability** | Automated inference logging and data drift detection using KS tests. |

---

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

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML Framework** | Scikit-learn |
| **MLOps/Tracking** | MLflow |
| **API Framework** | FastAPI + Uvicorn |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Deployment** | Docker |
| **DevOps** | GitHub Actions (CI/CD) |

---

## 📂 Project Structure

```text
Production_ML_Pipeline_with_MLOps/
├── app/                  # FastAPI production API
├── src/components/       # Modular ML components
│   ├── data_ingestion.py      # Data loading & validation
│   ├── data_preprocessing.py  # Feature engineering pipelines
│   ├── model_trainer.py       # Training, tuning & MLflow logging
│   ├── model_evaluation.py    # Test evaluation, reports & plots
│   ├── model_registry.py      # MLflow registry operations
│   └── monitoring.py          # Inference logging & drift detection
├── models/artifacts/     # Serialized model/preprocessor objects
├── reports/evaluation/   # Generated performance metrics & plots
├── logs/                 # Operational monitoring logs
├── Dockerfile            # Container definition
└── requirements.txt      # Project dependencies
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/Production_ML_Pipeline_with_MLOps.git
cd Production_ML_Pipeline_with_MLOps

pip install -r requirements.txt
```

### 2. Running the API
```bash
# Start the FastAPI service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

### 3. Docker Deployment
```bash
# Build the image
docker build -t production-ml-pipeline .

# Run the container
docker run -p 8000:8000 production-ml-pipeline
```

---

## 🧩 Components

*   **DataIngestion**: Handles dataset sourcing, quality validation, and splits (train/val/test).
*   **DataPreprocessing**: Constructs `ColumnTransformer` pipelines, ensuring consistent transformations across environments.
*   **ModelTrainer**: Handles multi-model comparison, hyperparameter optimization, and experiment tracking via MLflow.
*   **ModelEvaluator**: Generates metrics, confusion matrices, ROC/PR curves, and assesses model promotion status.
*   **ModelRegistry**: Interfaces with MLflow to promote models from Staging to Production.
*   **ModelMonitor**: Tracks live inference performance and calculates data drift.

---

## 🔄 MLOps Lifecycle

1.  **Develop**: Define components and data schema.
2.  **Train & Track**: Execute training pipeline; track all parameters, metrics, and models in MLflow.
3.  **Evaluate**: Automatically compare models; promote best model to *Staging* in Registry.
4.  **Promote**: Once verified, transition to *Production* stage.
5.  **Deploy**: FastAPI service automatically picks up the latest *Production* model version.
6.  **Monitor**: Real-time logging of predictions and drift detection to trigger retraining cycles.

---

<div align="center">

Built with ❤️ for scalable and reliable Machine Learning.

**⭐ Star this repo if it helped you!**

</div>
