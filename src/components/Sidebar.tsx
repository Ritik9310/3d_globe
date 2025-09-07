import React from 'react';
import { DebrisObject, SpacecraftObject } from '../types';
import { 
  Activity, 
  Compass, 
  Gauge, 
  MapPin, 
  Clock, 
  AlertTriangle,
  Satellite,
  Navigation,
  Info,
  BarChart3
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

interface SidebarProps {
  selectedObject?: DebrisObject | SpacecraftObject | null;
  onClose: () => void;
  predictions?: {
    collisionProbability: number;
    nextCloseApproach?: Date;
    riskFactors: string[];
    recommendations: string[];
  };
}

const Sidebar: React.FC<SidebarProps> = ({ selectedObject, onClose, predictions }) => {
  if (!selectedObject) return null;

  const isDebris = 'riskLevel' in selectedObject;
  const isSpacecraft = 'type' in selectedObject && !isDebris;

  const formatDistance = (value: number) => `${value.toFixed(1)} km`;
  const formatVelocity = (velocity: { x: number; y: number; z: number }) => {
    const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);
    return `${speed.toFixed(2)} km/s`;
  };

  const getRiskColor = (riskLevel?: string) => {
    switch (riskLevel) {
      case 'critical': return 'text-red-400';
      case 'high': return 'text-orange-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getRiskBadge = (riskLevel?: string) => {
    switch (riskLevel) {
      case 'critical': return 'bg-red-900/30 border-red-500/50 text-red-400';
      case 'high': return 'bg-orange-900/30 border-orange-500/50 text-orange-400';
      case 'medium': return 'bg-yellow-900/30 border-yellow-500/50 text-yellow-400';
      case 'low': return 'bg-green-900/30 border-green-500/50 text-green-400';
      default: return 'bg-gray-900/30 border-gray-500/50 text-gray-400';
    }
  };

  return (
    <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            {isDebris ? (
              <Activity className="w-6 h-6 text-red-400" />
            ) : (
              <Satellite className="w-6 h-6 text-green-400" />
            )}
            <div>
              <h2 className="text-xl font-semibold text-white">{selectedObject.name}</h2>
              <p className="text-gray-400 font-mono text-sm">{selectedObject.id}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1"
          >
            ✕
          </button>
        </div>
        
        {isDebris && (
          <div className="mt-4">
            <span className={`px-3 py-1.5 rounded-full text-sm font-medium border ${getRiskBadge((selectedObject as DebrisObject).riskLevel)}`}>
              {(selectedObject as DebrisObject).riskLevel.toUpperCase()} RISK
            </span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-6 space-y-6 max-h-[calc(100vh-200px)] overflow-y-auto">
        {/* Basic Information */}
        <div>
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <Info className="w-4 h-4" />
            Object Details
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Type:</span>
              <span className="text-white font-mono">
                {isSpacecraft ? (selectedObject as SpacecraftObject).type.toUpperCase() : 'Space Debris'}
              </span>
            </div>
            
            {isDebris && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-400">Size:</span>
                  <span className="text-white font-mono">
                    {((selectedObject as DebrisObject).size * 100).toFixed(1)} cm
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mass:</span>
                  <span className="text-white font-mono">
                    {((selectedObject as DebrisObject).mass).toFixed(1)} kg
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Risk Score:</span>
                  <span className={`font-mono font-bold ${getRiskColor((selectedObject as DebrisObject).riskLevel)}`}>
                    {(selectedObject as DebrisObject).riskScore}/100
                  </span>
                </div>
              </>
            )}

            {isSpacecraft && (
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`font-mono ${(selectedObject as SpacecraftObject).isActive ? 'text-green-400' : 'text-red-400'}`}>
                  {(selectedObject as SpacecraftObject).isActive ? 'Active' : 'Inactive'}
                </span>
              </div>
            )}

            <div className="flex justify-between">
              <span className="text-gray-400">Last Update:</span>
              <span className="text-white font-mono text-sm">
                {isDebris 
                  ? formatDistanceToNow((selectedObject as DebrisObject).lastUpdate) + ' ago'
                  : 'Real-time'
                }
              </span>
            </div>
          </div>
        </div>

        {/* Position and Velocity */}
        <div>
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <MapPin className="w-4 h-4" />
            Current State
          </h3>
          <div className="space-y-4">
            <div>
              <span className="text-gray-400 block mb-2">Position (km):</span>
              <div className="bg-black/30 rounded-lg p-3 font-mono text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">X:</span>
                  <span className="text-white">{selectedObject.position.x.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Y:</span>
                  <span className="text-white">{selectedObject.position.y.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Z:</span>
                  <span className="text-white">{selectedObject.position.z.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div>
              <span className="text-gray-400 block mb-2">Velocity (km/s):</span>
              <div className="bg-black/30 rounded-lg p-3 font-mono text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Vx:</span>
                  <span className="text-white">{selectedObject.velocity.x.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Vy:</span>
                  <span className="text-white">{selectedObject.velocity.y.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Vz:</span>
                  <span className="text-white">{selectedObject.velocity.z.toFixed(3)}</span>
                </div>
                <div className="flex justify-between pt-2 border-t border-white/10 mt-2">
                  <span className="text-gray-400">Speed:</span>
                  <span className="text-blue-400 font-bold">{formatVelocity(selectedObject.velocity)}</span>
                </div>
              </div>
            </div>

            <div className="flex justify-between">
              <span className="text-gray-400">Altitude:</span>
              <span className="text-white font-mono">{formatDistance(selectedObject.altitude)}</span>
            </div>
          </div>
        </div>

        {/* Orbital Parameters */}
        {isDebris && (selectedObject as DebrisObject).tle && (
          <div>
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <Compass className="w-4 h-4" />
              Orbital Elements
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Inclination:</span>
                <span className="text-white font-mono">
                  {(selectedObject as DebrisObject).inclination.toFixed(2)}°
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Eccentricity:</span>
                <span className="text-white font-mono">
                  {(selectedObject as DebrisObject).eccentricity.toFixed(4)}
                </span>
              </div>
              
              {/* TLE Data */}
              <div className="mt-4">
                <span className="text-gray-400 block mb-2">TLE Data:</span>
                <div className="bg-black/40 rounded-lg p-3 font-mono text-xs">
                  <div className="text-gray-300 mb-1">{(selectedObject as DebrisObject).tle!.line1}</div>
                  <div className="text-gray-300">{(selectedObject as DebrisObject).tle!.line2}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Close Approach Information */}
        {isDebris && (selectedObject as DebrisObject).nextCloseApproach && (
          <div>
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-yellow-400" />
              Next Close Approach
            </h3>
            <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-yellow-300">Target:</span>
                  <span className="text-white font-mono">
                    {(selectedObject as DebrisObject).nextCloseApproach!.targetId}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-yellow-300">Distance:</span>
                  <span className="text-yellow-400 font-mono font-bold">
                    {formatDistance((selectedObject as DebrisObject).nextCloseApproach!.distance)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-yellow-300">Time:</span>
                  <span className="text-white font-mono text-sm">
                    {format((selectedObject as DebrisObject).nextCloseApproach!.time, 'MMM dd, HH:mm UTC')}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Predictions */}
        {predictions && (
          <div>
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-blue-400" />
              AI Risk Assessment
            </h3>
            <div className="space-y-4">
              <div className="bg-black/30 rounded-lg p-4">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-gray-400">Collision Probability:</span>
                  <span className={`font-mono font-bold text-lg ${
                    predictions.collisionProbability > 0.5 ? 'text-red-400' :
                    predictions.collisionProbability > 0.2 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {(predictions.collisionProbability * 100).toFixed(2)}%
                  </span>
                </div>
                
                {/* Progress bar */}
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${
                      predictions.collisionProbability > 0.5 ? 'bg-red-500' :
                      predictions.collisionProbability > 0.2 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${Math.min(predictions.collisionProbability * 100, 100)}%` }}
                  />
                </div>
              </div>

              {predictions.nextCloseApproach && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Next CA Predicted:</span>
                  <span className="text-white font-mono text-sm">
                    {format(predictions.nextCloseApproach, 'MMM dd, HH:mm')}
                  </span>
                </div>
              )}

              {predictions.riskFactors.length > 0 && (
                <div>
                  <span className="text-red-400 block mb-2 font-medium">Risk Factors:</span>
                  <ul className="space-y-1">
                    {predictions.riskFactors.map((factor, index) => (
                      <li key={index} className="text-sm text-gray-300 flex items-start gap-2">
                        <span className="text-red-400 mt-0.5">•</span>
                        {factor}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {predictions.recommendations.length > 0 && (
                <div>
                  <span className="text-blue-400 block mb-2 font-medium">Recommendations:</span>
                  <ul className="space-y-1">
                    {predictions.recommendations.map((rec, index) => (
                      <li key={index} className="text-sm text-gray-300 flex items-start gap-2">
                        <span className="text-blue-400 mt-0.5">•</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="space-y-2">
          <button className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2">
            <Navigation className="w-4 h-4" />
            Track Object
          </button>
          
          {isSpacecraft && (
            <button className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2">
              <Clock className="w-4 h-4" />
              View Mission Timeline
            </button>
          )}
          
          {isDebris && (selectedObject as DebrisObject).riskLevel !== 'low' && (
            <button className="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Generate Avoidance Plan
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;