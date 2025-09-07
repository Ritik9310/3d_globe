export interface DebrisObject {
  id: string;
  name: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  velocity: {
    x: number;
    y: number;
    z: number;
  };
  size: number; // in meters
  mass: number; // in kg
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskScore: number; // 0-100
  altitude: number; // in km
  inclination: number; // in degrees
  eccentricity: number;
  lastUpdate: Date;
  nextCloseApproach?: {
    targetId: string;
    distance: number; // in km
    time: Date;
  };
  tle?: {
    line1: string;
    line2: string;
  };
}

export interface SpacecraftObject {
  id: string;
  name: string;
  type: 'ISS' | 'satellite' | 'spacecraft';
  position: {
    x: number;
    y: number;
    z: number;
  };
  velocity: {
    x: number;
    y: number;
    z: number;
  };
  altitude: number;
  isActive: boolean;
}

export interface CollisionAlert {
  id: string;
  debrisId: string;
  targetId: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  estimatedDistance: number; // in km
  timeToClosestApproach: number; // in hours
  probability: number; // 0-1
  suggestedManeuver?: {
    deltaV: number; // in m/s
    direction: string;
    executionTime: Date;
  };
  createdAt: Date;
}

export interface OrbitData {
  positions: Array<{
    x: number;
    y: number;
    z: number;
    time: number;
  }>;
  period: number; // in minutes
  apogee: number; // in km
  perigee: number; // in km
}

export interface SystemSettings {
  theme: 'dark' | 'light';
  units: 'metric' | 'imperial';
  alertThreshold: number; // in km
  simulationSpeed: number; // 1x, 2x, 5x, 10x
  showOrbits: boolean;
  showISS: boolean;
  showSatellites: boolean;
  autoRotate: boolean;
  notifications: boolean;
}

export interface PredictionResult {
  debrisId: string;
  predictedPositions: Array<{
    time: Date;
    position: { x: number; y: number; z: number };
    confidence: number;
  }>;
  riskAssessment: {
    collisionProbability: number;
    riskFactors: string[];
    recommendations: string[];
  };
}