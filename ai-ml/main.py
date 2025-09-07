"""
Space Debris Collision Prediction API
LSTM-based orbit prediction and risk assessment system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from models.orbit_predictor import OrbitPredictor
from models.collision_detector import CollisionDetector
from models.risk_assessor import RiskAssessor
from utils.tle_parser import TLEParser
from utils.space_weather import SpaceWeatherAPI
from utils.preprocessing import DataPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Space Debris AI/ML API",
    description="Advanced AI system for space debris tracking and collision prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI models
orbit_predictor = OrbitPredictor()
collision_detector = CollisionDetector()
risk_assessor = RiskAssessor()
tle_parser = TLEParser()
space_weather_api = SpaceWeatherAPI()
preprocessor = DataPreprocessor()

# Pydantic models for API
class DebrisInput(BaseModel):
    id: str
    position: Dict[str, float]  # x, y, z coordinates
    velocity: Dict[str, float]  # vx, vy, vz
    size: float
    mass: float
    altitude: float
    inclination: float
    eccentricity: float
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

class PredictionRequest(BaseModel):
    debris_objects: List[DebrisInput]
    spacecraft_positions: List[Dict]
    prediction_hours: int = 24
    include_uncertainties: bool = True
    space_weather_factor: bool = True

class PredictionResponse(BaseModel):
    debris_id: str
    predicted_positions: List[Dict]
    collision_probability: float
    risk_level: str
    risk_factors: List[str]
    recommendations: List[str]
    uncertainty_bounds: Dict[str, float]
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize models and load pre-trained weights"""
    print("Initializing AI/ML models...")
    
    # Load pre-trained models
    await orbit_predictor.initialize()
    await collision_detector.initialize()
    await risk_assessor.initialize()
    
    # Load space weather data
    await space_weather_api.initialize()
    
    print("All AI/ML models initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "models_loaded": {
            "orbit_predictor": orbit_predictor.is_initialized,
            "collision_detector": collision_detector.is_initialized,
            "risk_assessor": risk_assessor.is_initialized
        }
    }

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_collisions(request: PredictionRequest) -> List[PredictionResponse]:
    """
    Main endpoint for debris collision prediction
    Uses LSTM models to predict future positions and assess collision risks
    """
    start_time = datetime.utcnow()
    
    try:
        results = []
        
        # Get current space weather conditions if requested
        space_weather = None
        if request.space_weather_factor:
            space_weather = await space_weather_api.get_current_conditions()
        
        # Process each debris object
        for debris in request.debris_objects:
            # Preprocess TLE data if available
            orbital_elements = None
            if debris.tle_line1 and debris.tle_line2:
                orbital_elements = tle_parser.parse_tle(
                    debris.tle_line1, 
                    debris.tle_line2
                )
            
            # Prepare input features for ML models
            input_features = preprocessor.prepare_debris_features(
                debris, 
                space_weather, 
                orbital_elements
            )
            
            # Predict future positions using LSTM
            predicted_positions = await orbit_predictor.predict_orbit(
                input_features,
                hours_ahead=request.prediction_hours,
                include_uncertainties=request.include_uncertainties
            )
            
            # Detect collision probabilities
            collision_data = await collision_detector.assess_collisions(
                predicted_positions,
                request.spacecraft_positions,
                debris.size
            )
            
            # Perform risk assessment
            risk_assessment = await risk_assessor.evaluate_risk(
                debris,
                collision_data,
                space_weather
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            results.append(PredictionResponse(
                debris_id=debris.id,
                predicted_positions=predicted_positions,
                collision_probability=collision_data["probability"],
                risk_level=risk_assessment["level"],
                risk_factors=risk_assessment["factors"],
                recommendations=risk_assessment["recommendations"],
                uncertainty_bounds=collision_data.get("uncertainty_bounds", {}),
                processing_time=processing_time
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/orbit-propagation")
async def propagate_orbit(debris: DebrisInput, hours: int = 24):
    """
    Propagate a single object's orbit using advanced orbital mechanics
    """
    try:
        if not debris.tle_line1 or not debris.tle_line2:
            raise HTTPException(status_code=400, detail="TLE data required for orbit propagation")
        
        orbital_elements = tle_parser.parse_tle(debris.tle_line1, debris.tle_line2)
        
        # Get space weather effects
        space_weather = await space_weather_api.get_current_conditions()
        
        # Propagate orbit with perturbations
        propagated_orbit = await orbit_predictor.propagate_orbit_with_perturbations(
            orbital_elements,
            debris.size,
            debris.mass,
            hours,
            space_weather
        )
        
        return {
            "debris_id": debris.id,
            "orbit_data": propagated_orbit,
            "space_weather_effects": space_weather,
            "propagation_hours": hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orbit propagation failed: {str(e)}")

@app.get("/space-weather")
async def get_space_weather():
    """Get current space weather conditions affecting orbital mechanics"""
    try:
        conditions = await space_weather_api.get_current_conditions()
        forecast = await space_weather_api.get_forecast(days=7)
        
        return {
            "current_conditions": conditions,
            "7_day_forecast": forecast,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Space weather data unavailable: {str(e)}")

@app.post("/retrain-models")
async def retrain_models(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with latest data
    This would typically be run periodically
    """
    background_tasks.add_task(retrain_models_background)
    return {"message": "Model retraining initiated", "estimated_time": "2-4 hours"}

async def retrain_models_background():
    """Background task for model retraining"""
    try:
        print("Starting model retraining...")
        
        # Fetch latest TLE data and collision events
        training_data = await fetch_training_data()
        
        # Retrain orbit predictor
        await orbit_predictor.retrain(training_data["orbital_data"])
        
        # Retrain collision detector
        await collision_detector.retrain(training_data["collision_data"])
        
        # Retrain risk assessor
        await risk_assessor.retrain(training_data["risk_data"])
        
        print("Model retraining completed successfully")
        
    except Exception as e:
        print(f"Model retraining failed: {e}")

async def fetch_training_data():
    """Fetch latest training data from various sources"""
    # In a real implementation, this would fetch from:
    # - Space-Track.org for TLE data
    # - NASA's ODPO for collision events
    # - ESA's Space Debris Office
    # - NOAA for space weather data
    
    return {
        "orbital_data": [],
        "collision_data": [],
        "risk_data": []
    }

@app.get("/model-performance")
async def get_model_performance():
    """Get current model performance metrics"""
    try:
        orbit_metrics = await orbit_predictor.get_performance_metrics()
        collision_metrics = await collision_detector.get_performance_metrics()
        risk_metrics = await risk_assessor.get_performance_metrics()
        
        return {
            "orbit_predictor": orbit_metrics,
            "collision_detector": collision_metrics,
            "risk_assessor": risk_metrics,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve metrics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )