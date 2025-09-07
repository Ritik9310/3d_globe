import React, { useState, useEffect } from 'react';
import { io, Socket } from 'socket.io-client';
import Globe from './components/Globe';
import Controls from './components/Controls';
import Alerts from './components/Alerts';
import Sidebar from './components/Sidebar';
import { 
  DebrisObject, 
  SpacecraftObject, 
  CollisionAlert, 
  SystemSettings,
  PredictionResult
} from './types';
import { MOCK_DEBRIS_DATA, SPACECRAFT_DATA, TLEDataFetcher } from './data/debrisData';
import { CollisionDetectionSystem } from './utils/collisionDetection';

const INITIAL_SETTINGS: SystemSettings = {
  theme: 'dark',
  units: 'metric',
  alertThreshold: 25,
  simulationSpeed: 1,
  showOrbits: true,
  showISS: true,
  showSatellites: true,
  autoRotate: true,
  notifications: true,
};

function App() {
  const [debris, setDebris] = useState<DebrisObject[]>(MOCK_DEBRIS_DATA);
  const [spacecraft, setSpacecraft] = useState<SpacecraftObject[]>(SPACECRAFT_DATA);
  const [alerts, setAlerts] = useState<CollisionAlert[]>([]);
  const [settings, setSettings] = useState<SystemSettings>(INITIAL_SETTINGS);
  const [selectedObject, setSelectedObject] = useState<DebrisObject | SpacecraftObject | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult['riskAssessment'] | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [simulationSpeed, setSimulationSpeed] = useState(1);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [debugMode, setDebugMode] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io('ws://localhost:3001', {
      autoConnect: true,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    newSocket.on('connect', () => {
      console.log('Connected to debris tracking server');
    });

    newSocket.on('debris-update', (updatedDebris: DebrisObject[]) => {
      setDebris(updatedDebris);
      setLastUpdate(new Date());
    });

    newSocket.on('collision-alert', (alert: CollisionAlert) => {
      setAlerts(prev => [...prev, alert]);
      
      if (settings.notifications) {
        // Show browser notification
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification(`Collision Alert - ${alert.riskLevel.toUpperCase()}`, {
            body: `${alert.debrisId} approaching ${alert.targetId}`,
            icon: '/favicon.ico'
          });
        }
      }
    });

    newSocket.on('prediction-result', (result: PredictionResult) => {
      if (selectedObject && result.debrisId === selectedObject.id) {
        setPredictions(result.riskAssessment);
      }
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [settings.notifications, selectedObject]);

  // Request notification permission
  useEffect(() => {
    if ('Notification' in window && settings.notifications) {
      Notification.requestPermission();
    }
  }, [settings.notifications]);

  // Simulation loop for real-time updates
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      // Update debris positions (simplified simulation)
      setDebris(prevDebris => 
        prevDebris.map(d => {
          // Simple orbital motion simulation
          const timeStep = simulationSpeed * 0.1; // 0.1 seconds per frame
          const orbitalVelocity = Math.sqrt(398600.4418 / (d.altitude + 6371));
          const angularVelocity = orbitalVelocity / (d.altitude + 6371);
          
          const newPosition = {
            x: d.position.x * Math.cos(angularVelocity * timeStep) - d.position.y * Math.sin(angularVelocity * timeStep),
            y: d.position.x * Math.sin(angularVelocity * timeStep) + d.position.y * Math.cos(angularVelocity * timeStep),
            z: d.position.z
          };

          return {
            ...d,
            position: newPosition,
            lastUpdate: new Date()
          };
        })
      );

      setLastUpdate(new Date());
    }, 1000 / simulationSpeed); // Adjust frame rate based on simulation speed

    return () => clearInterval(interval);
  }, [isPlaying, simulationSpeed]);

  // Collision detection
  useEffect(() => {
    const interval = setInterval(() => {
      const newAlerts = CollisionDetectionSystem.detectImmediateThreats(
        debris, 
        spacecraft, 
        settings.alertThreshold
      );
      
      // Add new alerts, avoid duplicates
      setAlerts(prevAlerts => {
        const existingIds = new Set(prevAlerts.map(a => a.id));
        const uniqueNewAlerts = newAlerts.filter(a => !existingIds.has(a.id));
        return [...prevAlerts, ...uniqueNewAlerts];
      });

      // Generate future collision predictions
      const futureAlerts = CollisionDetectionSystem.predictFutureCollisions(debris, spacecraft, 24);
      if (futureAlerts.length > 0 && debugMode) {
        console.log(`Predicted ${futureAlerts.length} potential future collisions`);
      }
    }, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, [debris, spacecraft, settings.alertThreshold, debugMode]);

  // Update TLE data periodically
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!isPlaying) return;

      // Update a few debris objects with live TLE data
      const toUpdate = debris.slice(0, 10); // Update first 10 objects
      const updatedDebris = await Promise.all(
        toUpdate.map(d => TLEDataFetcher.updateDebrisWithLiveTLE(d))
      );

      setDebris(prevDebris => 
        prevDebris.map(d => {
          const updated = updatedDebris.find(u => u.id === d.id);
          return updated || d;
        })
      );
    }, 300000); // Every 5 minutes

    return () => clearInterval(interval);
  }, [debris, isPlaying]);

  // Get AI predictions for selected object
  useEffect(() => {
    if (!selectedObject || !('riskLevel' in selectedObject)) {
      setPredictions(null);
      return;
    }

    // Mock AI prediction API call
    const fetchPredictions = async () => {
      try {
        // In a real application, this would be an API call to your ML backend
        const mockPrediction = {
          collisionProbability: Math.random() * 0.3,
          nextCloseApproach: new Date(Date.now() + Math.random() * 7 * 24 * 60 * 60 * 1000),
          riskFactors: [
            'High atmospheric drag at current altitude',
            'Orbit precession due to J2 perturbations',
            'Solar radiation pressure effects'
          ].slice(0, Math.floor(Math.random() * 3) + 1),
          recommendations: [
            'Continue monitoring with enhanced tracking',
            'Consider debris mitigation strategies',
            'Coordinate with space situational awareness networks'
          ].slice(0, Math.floor(Math.random() * 2) + 1)
        };

        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        setPredictions(mockPrediction);
      } catch (error) {
        console.error('Failed to fetch predictions:', error);
      }
    };

    fetchPredictions();
  }, [selectedObject]);

  const handleSettingsChange = (newSettings: Partial<SystemSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };

  const handleObjectSelect = (object: DebrisObject | SpacecraftObject) => {
    setSelectedObject(object);
  };

  const handleDismissAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(a => a.id !== alertId));
  };

  const handleExecuteManeuver = (alert: CollisionAlert) => {
    console.log('Executing maneuver for alert:', alert.id);
    // In a real system, this would send commands to spacecraft
    
    // Update spacecraft position to simulate maneuver
    setSpacecraft(prev => 
      prev.map(s => {
        if (s.id === alert.targetId) {
          return {
            ...s,
            position: {
              x: s.position.x + (Math.random() - 0.5) * 10,
              y: s.position.y + (Math.random() - 0.5) * 10,
              z: s.position.z + (Math.random() - 0.5) * 5
            }
          };
        }
        return s;
      })
    );

    // Remove the alert after successful maneuver
    setTimeout(() => {
      handleDismissAlert(alert.id);
    }, 2000);
  };

  const filteredDebris = debris.filter(d => {
    // Filter based on settings
    return true; // For now, show all debris
  });

  const filteredSpacecraft = spacecraft.filter(s => {
    if (!settings.showISS && s.type === 'ISS') return false;
    if (!settings.showSatellites && s.type === 'satellite') return false;
    return true;
  });

  return (
    <div className={`min-h-screen ${settings.theme === 'dark' ? 'bg-black' : 'bg-gray-100'}`}>
      {/* Main Layout */}
      <div className="flex h-screen">
        {/* Left Panel - Controls */}
        <div className="w-80 p-4 bg-black/10 backdrop-blur-sm border-r border-white/10 overflow-y-auto">
          <Controls
            settings={settings}
            onSettingsChange={handleSettingsChange}
            isPlaying={isPlaying}
            onPlayPause={() => setIsPlaying(!isPlaying)}
            simulationSpeed={simulationSpeed}
            onSpeedChange={setSimulationSpeed}
            debugMode={debugMode}
            onToggleDebug={() => setDebugMode(!debugMode)}
          />
        </div>

        {/* Center - Globe */}
        <div className="flex-1 relative">
          <Globe
            debris={filteredDebris}
            spacecraft={filteredSpacecraft}
            settings={settings}
            selectedObject={selectedObject}
            onObjectSelect={handleObjectSelect}
          />
          
          {/* Status Bar */}
          <div className="absolute bottom-4 right-4 bg-black/30 backdrop-blur-sm rounded-lg px-4 py-2 border border-white/10">
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${socket?.connected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-gray-300">
                  {socket?.connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <span className="text-gray-400">|</span>
              <span className="text-gray-300 font-mono">
                Last update: {lastUpdate.toLocaleTimeString()}
              </span>
              {debugMode && (
                <>
                  <span className="text-gray-400">|</span>
                  <span className="text-orange-400 font-mono text-xs">DEBUG</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Alerts and Details */}
        <div className="w-96 p-4 bg-black/10 backdrop-blur-sm border-l border-white/10 space-y-4 overflow-y-auto">
          <Alerts
            alerts={alerts}
            onDismissAlert={handleDismissAlert}
            onExecuteManeuver={handleExecuteManeuver}
          />
          
          {selectedObject && (
            <Sidebar
              selectedObject={selectedObject}
              onClose={() => setSelectedObject(null)}
              predictions={predictions}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;