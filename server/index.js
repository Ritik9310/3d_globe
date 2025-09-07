const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Mock AI/ML prediction service
const mockMLService = {
  async predictCollisions(debrisData) {
    // Simulate ML processing time
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      predictions: debrisData.map(debris => ({
        debrisId: debris.id,
        predictedPositions: this.generateFuturePositions(debris, 24),
        riskAssessment: {
          collisionProbability: Math.random() * 0.1,
          riskFactors: this.generateRiskFactors(),
          recommendations: this.generateRecommendations()
        }
      }))
    };
  },

  generateFuturePositions(debris, hours) {
    const positions = [];
    const timeStep = 3600; // 1 hour steps
    
    for (let i = 0; i < hours; i++) {
      // Simplified orbital propagation
      const time = new Date(Date.now() + i * timeStep * 1000);
      const angle = (i * 0.1) % (2 * Math.PI);
      
      positions.push({
        time,
        position: {
          x: debris.position.x + Math.sin(angle) * 100,
          y: debris.position.y + Math.cos(angle) * 100,
          z: debris.position.z + Math.sin(angle * 0.5) * 50
        },
        confidence: Math.max(0.5, 1 - (i / hours) * 0.4)
      });
    }
    
    return positions;
  },

  generateRiskFactors() {
    const factors = [
      'High atmospheric drag coefficient',
      'Solar radiation pressure effects',
      'Orbit precession due to Earth\'s oblateness',
      'Atmospheric density variations',
      'Third-body perturbations from Moon/Sun',
      'Magnetospheric drag effects',
      'Attitude instability detected'
    ];
    
    return factors
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.floor(Math.random() * 4) + 1);
  },

  generateRecommendations() {
    const recommendations = [
      'Increase tracking frequency for 48 hours',
      'Coordinate with international space agencies',
      'Consider active debris removal mission',
      'Update orbital propagation models',
      'Enhance space weather monitoring',
      'Implement collision avoidance maneuver',
      'Deploy backup tracking systems'
    ];
    
    return recommendations
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.floor(Math.random() * 3) + 1);
  }
};

// Mock space weather service
const spaceWeatherService = {
  async getCurrentConditions() {
    return {
      solarActivity: 'moderate',
      geomagneticStorm: 'quiet',
      atmosphericDensity: 1.2e-12, // kg/m³ at 400km
      solarFluxIndex: 85 + Math.random() * 30,
      kpIndex: Math.random() * 3,
      timestamp: new Date()
    };
  }
};

// Store active debris data
let currentDebrisData = [];
let connectedClients = 0;

// API Routes
app.get('/api/debris', async (req, res) => {
  try {
    const { limit = 500, riskLevel, altitude } = req.query;
    
    // In a real system, this would fetch from a database or external API
    let filteredDebris = [...currentDebrisData];
    
    if (riskLevel) {
      filteredDebris = filteredDebris.filter(d => d.riskLevel === riskLevel);
    }
    
    if (altitude) {
      const [minAlt, maxAlt] = altitude.split('-').map(Number);
      filteredDebris = filteredDebris.filter(d => 
        d.altitude >= minAlt && d.altitude <= maxAlt
      );
    }
    
    filteredDebris = filteredDebris.slice(0, parseInt(limit));
    
    // Enrich with space weather data
    const spaceWeather = await spaceWeatherService.getCurrentConditions();
    
    res.json({
      debris: filteredDebris,
      metadata: {
        totalObjects: currentDebrisData.length,
        filteredCount: filteredDebris.length,
        lastUpdate: new Date(),
        spaceWeather
      }
    });
  } catch (error) {
    console.error('Error fetching debris data:', error);
    res.status(500).json({ error: 'Failed to fetch debris data' });
  }
});

app.post('/api/predict', async (req, res) => {
  try {
    const { debrisIds, timeWindow = 24 } = req.body;
    
    const targetDebris = currentDebrisData.filter(d => 
      debrisIds.includes(d.id)
    );
    
    const predictions = await mockMLService.predictCollisions(targetDebris);
    
    res.json({
      success: true,
      predictions: predictions.predictions,
      processedAt: new Date(),
      timeWindow
    });
  } catch (error) {
    console.error('Error generating predictions:', error);
    res.status(500).json({ error: 'Prediction service unavailable' });
  }
});

app.get('/api/space-weather', async (req, res) => {
  try {
    const conditions = await spaceWeatherService.getCurrentConditions();
    res.json(conditions);
  } catch (error) {
    console.error('Error fetching space weather:', error);
    res.status(500).json({ error: 'Space weather service unavailable' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date(),
    connectedClients,
    debrisObjects: currentDebrisData.length,
    uptime: process.uptime()
  });
});

// WebSocket handling
io.on('connection', (socket) => {
  connectedClients++;
  console.log(`Client connected. Total clients: ${connectedClients}`);
  
  // Send initial data
  socket.emit('debris-update', currentDebrisData);
  
  socket.on('disconnect', () => {
    connectedClients--;
    console.log(`Client disconnected. Total clients: ${connectedClients}`);
  });
  
  socket.on('request-predictions', async (data) => {
    try {
      const predictions = await mockMLService.predictCollisions(
        currentDebrisData.filter(d => data.debrisIds.includes(d.id))
      );
      
      predictions.predictions.forEach(prediction => {
        socket.emit('prediction-result', prediction);
      });
    } catch (error) {
      console.error('Error generating predictions:', error);
      socket.emit('error', { message: 'Prediction service error' });
    }
  });
});

// Simulation: Update debris positions and generate alerts
function simulateDebrisMovement() {
  if (currentDebrisData.length === 0) {
    // Initialize with mock data if empty
    currentDebrisData = require('./mockData').generateMockDebris(500);
  }
  
  // Update positions (simplified orbital mechanics)
  currentDebrisData = currentDebrisData.map(debris => {
    const timeStep = 60; // 1 minute
    const orbitalVelocity = Math.sqrt(398600.4418 / (debris.altitude + 6371));
    const angularVelocity = orbitalVelocity / (debris.altitude + 6371);
    
    // Simple rotation
    const angle = angularVelocity * timeStep;
    const newPosition = {
      x: debris.position.x * Math.cos(angle) - debris.position.y * Math.sin(angle),
      y: debris.position.x * Math.sin(angle) + debris.position.y * Math.cos(angle),
      z: debris.position.z + (Math.random() - 0.5) * 0.1 // Small random variation
    };
    
    return {
      ...debris,
      position: newPosition,
      lastUpdate: new Date()
    };
  });
  
  // Broadcast updates to all clients
  io.emit('debris-update', currentDebrisData);
  
  // Generate occasional collision alerts (for demo)
  if (Math.random() < 0.1) { // 10% chance per minute
    const randomDebris = currentDebrisData[Math.floor(Math.random() * currentDebrisData.length)];
    const alert = {
      id: `ALERT-${Date.now()}`,
      debrisId: randomDebris.id,
      targetId: 'ISS-ZARYA',
      riskLevel: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
      estimatedDistance: Math.random() * 50 + 5,
      timeToClosestApproach: Math.random() * 24,
      probability: Math.random() * 0.3,
      createdAt: new Date()
    };
    
    io.emit('collision-alert', alert);
  }
}

// Start simulation
setInterval(simulateDebrisMovement, 10000); // Update every 10 seconds

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`Space Debris Tracking Server running on port ${PORT}`);
  console.log(`WebSocket endpoint: ws://localhost:${PORT}`);
  console.log(`REST API endpoint: http://localhost:${PORT}/api`);
});