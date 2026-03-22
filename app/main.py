from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from src.components.model_registry import ModelRegistry, ModelRegistryConfig
from src.components.data_preprocessing import DataPreprocessing, PreprocessingConfig
from src.components.monitoring import ModelMonitor, MonitoringConfig

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production ML API")

# Load model and components
registry = ModelRegistry(ModelRegistryConfig())
preprocessor = DataPreprocessing(PreprocessingConfig())
monitor = ModelMonitor(MonitoringConfig())

# Load production model
try:
    prod_model, prod_version = registry.get_production_model()
    if prod_model:
        preprocessor.load_preprocessor()
        logger.info("Model and preprocessor loaded successfully")
    else:
        logger.warning("No production model found.")
except Exception as e:
    logger.error(f"Failed to load production artifacts: {e}")
    prod_model = None

class PredictionRequest(BaseModel):
    data: dict

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": prod_model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not prod_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([request.data])
        X_transformed, _ = preprocessor.transform(df)
        prediction = prod_model.predict(X_transformed)
        
        # Log for monitoring
        monitor.log_inference(df, prediction[0], "v1")
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
