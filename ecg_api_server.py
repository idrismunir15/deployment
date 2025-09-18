
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECG Abnormality Classification API",
    description="AI-powered ECG classification for detecting cardiac abnormalities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model_data = None

# Pydantic models for API
class ECGFeatures(BaseModel):
    class Config:
        @staticmethod
        def schema_extra():
            global model_data
            if model_data is None:
                try:
                    load_model()
                except Exception:
                    return {"example": {}}
            features = model_data['feature_selector_features'] if model_data else []
            example = {}
            for feat in features:
                if feat.lower() == 'gender':
                    example[feat] = 'M'
                elif feat.lower() == 'age':
                    example[feat] = 65
                elif 'qtc' in feat.lower():
                    example[feat] = 420.5
                elif 'qrs' in feat.lower():
                    example[feat] = 98.2
                else:
                    example[feat] = 0
            return {"example": example}

class ECGBatchRequest(BaseModel):
    """Batch ECG data input"""
    data: List[Dict[str, Any]] = Field(..., description="List of ECG feature dictionaries")
    threshold: Optional[float] = Field(0.5, description="Classification threshold")

class PredictionResponse(BaseModel):
    """Prediction response model"""
    probability: float = Field(..., description="Probability of abnormality (0-1)")
    prediction: str = Field(..., description="Normal or Abnormal")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    accuracy: float
    features_count: int
    training_samples: int

def load_model():
    """Load the trained ECG classification model"""
    global model_data
    try:
        # Try to load model file
        model_path = os.environ.get('MODEL_PATH', 'ecg_abnormality_classifier_lightgbm.pkl')
        model_data = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess ECG data using fitted transformers"""
    global model_data
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get fitted preprocessors
    numeric_imputer = model_data['numeric_imputer']
    categorical_imputer = model_data['categorical_imputer']
    label_encoders = model_data['label_encoders']
    numeric_cols = model_data['numeric_columns']
    categorical_cols = model_data['categorical_columns']
    
    # Create a copy to avoid modifying original data
    processed_data = data.copy()
    
    # Apply preprocessing using fitted transformers
    if numeric_imputer is not None and len(numeric_cols) > 0:
        available_numeric_cols = [col for col in numeric_cols if col in processed_data.columns]
        if available_numeric_cols:
            processed_data[available_numeric_cols] = numeric_imputer.transform(
                processed_data[available_numeric_cols]
            )
    
    if categorical_imputer is not None and len(categorical_cols) > 0:
        available_categorical_cols = [col for col in categorical_cols if col in processed_data.columns]
        if available_categorical_cols:
            processed_data[available_categorical_cols] = categorical_imputer.transform(
                processed_data[available_categorical_cols]
            )
            
            # Apply fitted label encoders
            for col in available_categorical_cols:
                if col in label_encoders:
                    le = label_encoders[col]
                    processed_data[col] = processed_data[col].astype(str)
                    unseen_mask = ~processed_data[col].isin(le.classes_)
                    if unseen_mask.any():
                        most_frequent_class = le.classes_[0]
                        processed_data.loc[unseen_mask, col] = most_frequent_class
                    processed_data[col] = le.transform(processed_data[col])
    
    # Select only the features that were used in training
    selected_features = model_data['feature_selector_features']
    available_features = [feat for feat in selected_features if feat in processed_data.columns]
    
    if len(available_features) != len(selected_features):
        missing_features = set(selected_features) - set(available_features)
        logger.warning(f"Missing features: {missing_features}")
    
    return processed_data[available_features]

def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ECG Abnormality Classification API",
        "status": "healthy",
        "docs": "/docs",
        "deployed_on": "Heroku"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "timestamp": pd.Timestamp.now().isoformat(),
        "platform": "heroku"
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    global model_data
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    metrics = model_data['performance_metrics']
    return ModelInfo(
        model_name="LightGBM ECG Classifier",
        version="1.0.0",
        accuracy=float(metrics['roc_auc']),
        features_count=len(model_data['feature_selector_features']),
        training_samples=5068
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: Dict[str, Any], threshold: float = 0.5):
    """Predict abnormality for single ECG sample"""
    global model_data
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        processed_data = preprocess_data(df)
        
        # Predict
        model = model_data['model']
        probability = float(model.predict_proba(processed_data)[0, 1])
        
        # Create response
        prediction = "Abnormal" if probability > threshold else "Normal"
        confidence = probability if prediction == "Abnormal" else (1 - probability)
        
        return PredictionResponse(
            probability=probability,
            prediction=prediction,
            confidence=confidence,
            risk_level=get_risk_level(probability)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: ECGBatchRequest):
    """Predict abnormality for batch of ECG samples"""
    global model_data
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Preprocess
        processed_data = preprocess_data(df)
        
        # Predict
        model = model_data['model']
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        # Create responses
        predictions = []
        abnormal_count = 0
        
        for prob in probabilities:
            prediction = "Abnormal" if prob > request.threshold else "Normal"
            confidence = prob if prediction == "Abnormal" else (1 - prob)
            
            if prediction == "Abnormal":
                abnormal_count += 1
                
            predictions.append(PredictionResponse(
                probability=float(prob),
                prediction=prediction,
                confidence=float(confidence),
                risk_level=get_risk_level(prob)
            ))
        
        # Summary statistics
        summary = {
            "total_samples": len(predictions),
            "abnormal_count": abnormal_count,
            "normal_count": len(predictions) - abnormal_count,
            "abnormal_percentage": round((abnormal_count / len(predictions)) * 100, 2),
            "average_probability": float(np.mean(probabilities)),
            "threshold_used": request.threshold
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), threshold: float = 0.5):
    """Predict abnormality from CSV file"""
    global model_data
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess and predict
        processed_data = preprocess_data(df)
        model = model_data['model']
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        # Add results to original DataFrame
        df['probability'] = probabilities
        df['prediction'] = ['Abnormal' if p > threshold else 'Normal' for p in probabilities]
        df['risk_level'] = [get_risk_level(p) for p in probabilities]
        
        # Convert to JSON for response
        results = df.to_dict('records')
        
        return {
            "results": results,
            "summary": {
                "total_samples": len(results),
                "abnormal_count": sum(1 for r in results if r['prediction'] == 'Abnormal'),
                "average_probability": float(np.mean(probabilities))
            }
        }
        
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"CSV prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)