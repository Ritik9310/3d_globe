"""
Advanced Collision Detection and Risk Assessment Model
Uses machine learning to predict collision probabilities and assess risks
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio

class CollisionDetector:
    def __init__(self):
        self.classifier = None  # For collision/no-collision classification
        self.probability_regressor = None  # For probability estimation
        self.scaler = None
        self.is_initialized = False
        self.feature_names = [
            'relative_distance', 'relative_velocity', 'approach_angle',
            'debris_size', 'spacecraft_size', 'time_to_ca',
            'position_uncertainty', 'velocity_uncertainty',
            'atmospheric_density', 'solar_activity', 'geomagnetic_index'
        ]
    
    async def initialize(self):
        """Initialize or load pre-trained collision detection models"""
        try:
            # Try to load existing models
            self.classifier = joblib.load('models/collision_classifier.pkl')
            self.probability_regressor = joblib.load('models/probability_regressor.pkl')
            self.scaler = joblib.load('models/collision_scaler.pkl')
            print("Loaded pre-trained collision detection models")
        except:
            # Create new models if none exist
            self.classifier = self._build_classifier()
            self.probability_regressor = self._build_probability_regressor()
            self.scaler = StandardScaler()
            print("Created new collision detection models")
        
        self.is_initialized = True
    
    def _build_classifier(self) -> RandomForestClassifier:
        """Build collision classification model"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def _build_probability_regressor(self) -> GradientBoostingRegressor:
        """Build collision probability regression model"""
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    
    async def assess_collisions(self,
                              predicted_positions: List[Dict],
                              spacecraft_positions: List[Dict],
                              debris_size: float) -> Dict:
        """
        Assess collision probabilities between debris and spacecraft
        
        Args:
            predicted_positions: Future positions of debris
            spacecraft_positions: Positions of spacecraft/satellites
            debris_size: Size of debris object in meters
        
        Returns:
            Collision assessment with probabilities and risk factors
        """
        if not self.is_initialized:
            raise RuntimeError("Collision detector not initialized")
        
        collision_events = []
        max_probability = 0.0
        critical_encounters = []
        
        # Check each predicted position against all spacecraft
        for debris_pos in predicted_positions:
            for spacecraft in spacecraft_positions:
                # Calculate encounter geometry
                encounter_data = self._calculate_encounter_geometry(
                    debris_pos, spacecraft, debris_size
                )
                
                if encounter_data['relative_distance'] < 100:  # Within 100km
                    # Extract features for ML model
                    features = self._extract_collision_features(encounter_data)
                    
                    # Predict collision probability
                    probability = self._predict_collision_probability(features)
                    
                    # Classify risk level
                    risk_level = self._classify_risk_level(features)
                    
                    collision_event = {
                        'time': debris_pos['time'],
                        'spacecraft_id': spacecraft.get('id', 'unknown'),
                        'relative_distance': encounter_data['relative_distance'],
                        'relative_velocity': encounter_data['relative_velocity'],
                        'collision_probability': probability,
                        'risk_level': risk_level,
                        'miss_distance': encounter_data['miss_distance'],
                        'approach_angle': encounter_data['approach_angle']
                    }
                    
                    collision_events.append(collision_event)
                    max_probability = max(max_probability, probability)
                    
                    if probability > 0.1:  # High risk threshold
                        critical_encounters.append(collision_event)
        
        # Calculate uncertainty bounds
        uncertainty_bounds = self._calculate_uncertainty_bounds(collision_events)
        
        return {
            'probability': max_probability,
            'collision_events': collision_events,
            'critical_encounters': critical_encounters,
            'uncertainty_bounds': uncertainty_bounds,
            'assessment_timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_encounter_geometry(self,
                                    debris_pos: Dict,
                                    spacecraft: Dict,
                                    debris_size: float) -> Dict:
        """Calculate geometric parameters of potential encounter"""
        
        # Extract positions
        debris_position = np.array([
            debris_pos['position']['x'],
            debris_pos['position']['y'],
            debris_pos['position']['z']
        ])
        
        spacecraft_position = np.array([
            spacecraft['position']['x'],
            spacecraft['position']['y'],
            spacecraft['position']['z']
        ])
        
        # Extract velocities
        debris_velocity = np.array([
            debris_pos['velocity']['x'],
            debris_pos['velocity']['y'],
            debris_pos['velocity']['z']
        ])
        
        spacecraft_velocity = np.array([
            spacecraft['velocity']['x'],
            spacecraft['velocity']['y'],
            spacecraft['velocity']['z']
        ])
        
        # Calculate relative vectors
        relative_position = debris_position - spacecraft_position
        relative_velocity = debris_velocity - spacecraft_velocity
        
        # Calculate distances and speeds
        relative_distance = np.linalg.norm(relative_position)
        relative_speed = np.linalg.norm(relative_velocity)
        
        # Calculate miss distance (closest approach distance)
        if relative_speed > 0:
            # Time to closest approach
            time_to_ca = -np.dot(relative_position, relative_velocity) / (relative_speed ** 2)
            time_to_ca = max(0, time_to_ca)  # Only future encounters
            
            # Position at closest approach
            ca_position = relative_position + relative_velocity * time_to_ca
            miss_distance = np.linalg.norm(ca_position)
        else:
            time_to_ca = 0
            miss_distance = relative_distance
        
        # Calculate approach angle
        if relative_speed > 0 and relative_distance > 0:
            cos_angle = np.dot(relative_position, relative_velocity) / (relative_distance * relative_speed)
            approach_angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        else:
            approach_angle = 0
        
        return {
            'relative_distance': relative_distance,
            'relative_velocity': relative_speed,
            'miss_distance': miss_distance,
            'time_to_ca': time_to_ca,
            'approach_angle': approach_angle,
            'spacecraft_size': spacecraft.get('size', 10),  # Default 10m
            'debris_size': debris_size
        }
    
    def _extract_collision_features(self, encounter_data: Dict) -> np.ndarray:
        """Extract features for collision prediction model"""
        
        # Get space weather data (mock for demo)
        atmospheric_density = 1e-12  # kg/m³
        solar_activity = 85  # F10.7 solar flux
        geomagnetic_index = 2.0  # Kp index
        
        # Position and velocity uncertainties (mock)
        position_uncertainty = 0.1  # km
        velocity_uncertainty = 0.001  # km/s
        
        features = np.array([
            encounter_data['relative_distance'],
            encounter_data['relative_velocity'],
            encounter_data['approach_angle'],
            encounter_data['debris_size'],
            encounter_data['spacecraft_size'],
            encounter_data['time_to_ca'],
            position_uncertainty,
            velocity_uncertainty,
            atmospheric_density,
            solar_activity,
            geomagnetic_index
        ])
        
        return features
    
    def _predict_collision_probability(self, features: np.ndarray) -> float:
        """Predict collision probability using trained models"""
        if self.probability_regressor is None:
            # Fallback analytical model
            return self._analytical_collision_probability(features)
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict probability
            probability = self.probability_regressor.predict(features_scaled)[0]
            
            # Ensure probability is in valid range
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            print(f"ML prediction failed: {e}, using analytical model")
            return self._analytical_collision_probability(features)
    
    def _analytical_collision_probability(self, features: np.ndarray) -> float:
        """Fallback analytical collision probability model"""
        relative_distance = features[0]  # km
        relative_velocity = features[1]   # km/s
        debris_size = features[3]         # m
        spacecraft_size = features[4]     # m
        
        # Combined cross-sectional area
        combined_radius = (debris_size + spacecraft_size) / 2000  # Convert to km
        cross_section = np.pi * (combined_radius ** 2)
        
        # Simple geometric probability
        if relative_distance < combined_radius:
            base_probability = 1.0
        else:
            base_probability = cross_section / (np.pi * relative_distance ** 2)
        
        # Velocity factor (higher relative velocity = less time for collision)
        velocity_factor = 1.0 / (1.0 + relative_velocity / 10.0)
        
        # Uncertainty factor
        uncertainty_factor = 1.5  # Account for position/velocity uncertainties
        
        probability = base_probability * velocity_factor * uncertainty_factor
        
        return min(1.0, probability)
    
    def _classify_risk_level(self, features: np.ndarray) -> str:
        """Classify encounter risk level"""
        relative_distance = features[0]
        debris_size = features[3]
        
        if relative_distance < 1.0 and debris_size > 0.1:
            return 'critical'
        elif relative_distance < 5.0 and debris_size > 0.05:
            return 'high'
        elif relative_distance < 25.0:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_uncertainty_bounds(self, collision_events: List[Dict]) -> Dict:
        """Calculate uncertainty bounds for collision predictions"""
        if not collision_events:
            return {'position': 0.0, 'velocity': 0.0, 'probability': 0.0}
        
        # Extract probabilities
        probabilities = [event['collision_probability'] for event in collision_events]
        
        # Calculate statistical measures
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        return {
            'position': 0.1,  # km (mock)
            'velocity': 0.001,  # km/s (mock)
            'probability': std_prob,
            'confidence_interval_95': {
                'lower': max(0, mean_prob - 1.96 * std_prob),
                'upper': min(1, mean_prob + 1.96 * std_prob)
            }
        }
    
    async def detect_conjunction_events(self,
                                      debris_catalog: List[Dict],
                                      spacecraft_catalog: List[Dict],
                                      time_window_hours: int = 72) -> List[Dict]:
        """
        Detect all potential conjunction events in the given time window
        """
        conjunction_events = []
        
        for debris in debris_catalog:
            for spacecraft in spacecraft_catalog:
                # Generate future positions for both objects
                debris_positions = await self._generate_positions(debris, time_window_hours)
                spacecraft_positions = await self._generate_positions(spacecraft, time_window_hours)
                
                # Find close approaches
                close_approaches = self._find_close_approaches(
                    debris_positions, 
                    spacecraft_positions,
                    threshold_km=100
                )
                
                for approach in close_approaches:
                    # Calculate detailed conjunction data
                    conjunction = await self._analyze_conjunction(
                        debris, spacecraft, approach
                    )
                    
                    if conjunction['miss_distance'] < 50:  # Within 50km
                        conjunction_events.append(conjunction)
        
        # Sort by collision probability
        conjunction_events.sort(key=lambda x: x['collision_probability'], reverse=True)
        
        return conjunction_events
    
    async def _generate_positions(self, space_object: Dict, hours: int) -> List[Dict]:
        """Generate future positions for a space object"""
        # This would use the orbit predictor
        # For demo, generate mock positions
        positions = []
        
        for hour in range(hours):
            # Simple orbital motion
            angle = hour * 0.1  # rad
            radius = space_object.get('altitude', 400) + 6371  # km
            
            position = {
                'time': (datetime.utcnow() + timedelta(hours=hour)).isoformat(),
                'position': {
                    'x': radius * np.cos(angle),
                    'y': radius * np.sin(angle),
                    'z': 0
                },
                'velocity': {
                    'x': -7.5 * np.sin(angle),
                    'y': 7.5 * np.cos(angle),
                    'z': 0
                }
            }
            positions.append(position)
        
        return positions
    
    def _find_close_approaches(self,
                              positions1: List[Dict],
                              positions2: List[Dict],
                              threshold_km: float) -> List[Dict]:
        """Find close approaches between two position tracks"""
        close_approaches = []
        
        for i, pos1 in enumerate(positions1):
            for j, pos2 in enumerate(positions2):
                # Check if times are close (within 1 hour)
                time1 = datetime.fromisoformat(pos1['time'].replace('Z', '+00:00'))
                time2 = datetime.fromisoformat(pos2['time'].replace('Z', '+00:00'))
                
                if abs((time1 - time2).total_seconds()) < 3600:  # 1 hour
                    distance = np.linalg.norm(np.array([
                        pos1['position']['x'] - pos2['position']['x'],
                        pos1['position']['y'] - pos2['position']['y'],
                        pos1['position']['z'] - pos2['position']['z']
                    ]))
                    
                    if distance < threshold_km:
                        close_approaches.append({
                            'time': pos1['time'],
                            'distance': distance,
                            'position1': pos1['position'],
                            'position2': pos2['position'],
                            'velocity1': pos1['velocity'],
                            'velocity2': pos2['velocity']
                        })
        
        return close_approaches
    
    async def _analyze_conjunction(self,
                                 debris: Dict,
                                 spacecraft: Dict,
                                 approach: Dict) -> Dict:
        """Analyze a specific conjunction event"""
        
        # Extract encounter geometry
        encounter_data = self._calculate_encounter_geometry_from_approach(
            approach, debris.get('size', 0.1)
        )
        
        # Extract features
        features = self._extract_collision_features(encounter_data)
        
        # Predict collision probability
        probability = self._predict_collision_probability(features)
        
        return {
            'debris_id': debris.get('id', 'unknown'),
            'spacecraft_id': spacecraft.get('id', 'unknown'),
            'time_of_closest_approach': approach['time'],
            'miss_distance': approach['distance'],
            'collision_probability': probability,
            'risk_level': self._classify_risk_level(features),
            'relative_velocity': encounter_data['relative_velocity'],
            'approach_geometry': encounter_data,
            'recommendation': self._generate_recommendation(probability, approach['distance'])
        }
    
    def _calculate_encounter_geometry_from_approach(self,
                                                   approach: Dict,
                                                   debris_size: float) -> Dict:
        """Calculate encounter geometry from approach data"""
        
        # Calculate relative velocity
        rel_vel = np.array([
            approach['velocity1']['x'] - approach['velocity2']['x'],
            approach['velocity1']['y'] - approach['velocity2']['y'],
            approach['velocity1']['z'] - approach['velocity2']['z']
        ])
        
        relative_speed = np.linalg.norm(rel_vel)
        
        return {
            'relative_distance': approach['distance'],
            'relative_velocity': relative_speed,
            'miss_distance': approach['distance'],
            'time_to_ca': 0,  # At closest approach
            'approach_angle': 90,  # Simplified
            'spacecraft_size': 10,  # Default
            'debris_size': debris_size
        }
    
    def _generate_recommendation(self, probability: float, miss_distance: float) -> str:
        """Generate recommendation based on collision risk"""
        if probability > 0.1:
            return "URGENT: Execute collision avoidance maneuver immediately"
        elif probability > 0.01:
            return "WARNING: Consider precautionary maneuver"
        elif miss_distance < 5.0:
            return "CAUTION: Monitor closely, prepare for potential maneuver"
        else:
            return "NOMINAL: Continue normal operations with routine monitoring"
    
    async def retrain(self, training_data: List[Dict]):
        """Retrain collision detection models with new data"""
        if not training_data:
            print("No training data provided for collision detector")
            return
        
        print(f"Retraining collision detector with {len(training_data)} samples")
        
        # Prepare training data
        X, y_class, y_prob = self._prepare_training_data(training_data)
        
        if len(X) < 50:
            print("Insufficient training data for collision detector")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classification model
        self.classifier.fit(X_scaled, y_class)
        
        # Train probability regression model
        self.probability_regressor.fit(X_scaled, y_prob)
        
        # Save models
        joblib.dump(self.classifier, 'models/collision_classifier.pkl')
        joblib.dump(self.probability_regressor, 'models/probability_regressor.pkl')
        joblib.dump(self.scaler, 'models/collision_scaler.pkl')
        
        print("Collision detector retrained successfully")
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for collision models"""
        # This would process historical collision/conjunction data
        # For demo, return mock training data
        n_samples = len(training_data)
        n_features = len(self.feature_names)
        
        X = np.random.randn(n_samples, n_features)
        y_class = np.random.randint(0, 2, n_samples)  # Binary collision/no-collision
        y_prob = np.random.random(n_samples) * 0.1    # Low collision probabilities
        
        return X, y_class, y_prob
    
    async def get_performance_metrics(self) -> Dict:
        """Get collision detector performance metrics"""
        return {
            'model_type': 'Random Forest + Gradient Boosting',
            'classification_accuracy': 0.95,  # Mock value
            'probability_rmse': 0.02,          # Mock value
            'false_positive_rate': 0.03,      # Mock value
            'false_negative_rate': 0.01,      # Mock value
            'features_used': self.feature_names,
            'last_retrained': datetime.utcnow().isoformat(),
            'training_samples': 5000,          # Mock value
            'validation_samples': 1000         # Mock value
        }