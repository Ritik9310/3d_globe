import React from 'react';
import { SystemSettings } from '../types';
import { 
  Settings, 
  Play, 
  Pause, 
  RotateCcw, 
  Zap,
  Eye,
  EyeOff,
  Moon,
  Sun,
  Gauge,
  Bell,
  BellOff
} from 'lucide-react';

interface ControlsProps {
  settings: SystemSettings;
  onSettingsChange: (settings: Partial<SystemSettings>) => void;
  isPlaying: boolean;
  onPlayPause: () => void;
  simulationSpeed: number;
  onSpeedChange: (speed: number) => void;
  debugMode?: boolean;
  onToggleDebug?: () => void;
}

const Controls: React.FC<ControlsProps> = ({
  settings,
  onSettingsChange,
  isPlaying,
  onPlayPause,
  simulationSpeed,
  onSpeedChange,
  debugMode = false,
  onToggleDebug
}) => {
  const speedOptions = [
    { value: 0.5, label: '0.5x' },
    { value: 1, label: '1x' },
    { value: 2, label: '2x' },
    { value: 5, label: '5x' },
    { value: 10, label: '10x' }
  ];

  return (
    <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Settings className="w-5 h-5 text-blue-400" />
        <h2 className="text-xl font-semibold text-white">Mission Control</h2>
      </div>

      {/* Simulation Controls */}
      <div className="space-y-6">
        {/* Play/Pause and Speed */}
        <div className="bg-black/30 rounded-lg p-4 border border-white/5">
          <h3 className="text-white font-medium mb-4 flex items-center gap-2">
            <Gauge className="w-4 h-4" />
            Simulation Control
          </h3>
          
          <div className="flex items-center gap-4 mb-4">
            <button
              onClick={onPlayPause}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all ${
                isPlaying 
                  ? 'bg-orange-600/20 border-orange-500/30 text-orange-400 hover:bg-orange-600/30' 
                  : 'bg-green-600/20 border-green-500/30 text-green-400 hover:bg-green-600/30'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm">Speed:</span>
              <select
                value={simulationSpeed}
                onChange={(e) => onSpeedChange(Number(e.target.value))}
                className="bg-black/40 border border-white/20 rounded px-2 py-1 text-white text-sm focus:outline-none focus:border-blue-500"
              >
                {speedOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="text-xs text-gray-500">
            Simulation running at {simulationSpeed}x real-time speed
          </div>
        </div>

        {/* Display Options */}
        <div className="bg-black/30 rounded-lg p-4 border border-white/5">
          <h3 className="text-white font-medium mb-4 flex items-center gap-2">
            <Eye className="w-4 h-4" />
            Display Options
          </h3>
          
          <div className="space-y-3">
            <label className="flex items-center justify-between cursor-pointer group">
              <span className="text-gray-300 group-hover:text-white transition-colors">
                Show Orbital Paths
              </span>
              <button
                onClick={() => onSettingsChange({ showOrbits: !settings.showOrbits })}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.showOrbits ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.showOrbits ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>
            
            <label className="flex items-center justify-between cursor-pointer group">
              <span className="text-gray-300 group-hover:text-white transition-colors">
                Show ISS
              </span>
              <button
                onClick={() => onSettingsChange({ showISS: !settings.showISS })}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.showISS ? 'bg-green-600' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.showISS ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>
            
            <label className="flex items-center justify-between cursor-pointer group">
              <span className="text-gray-300 group-hover:text-white transition-colors">
                Show Satellites
              </span>
              <button
                onClick={() => onSettingsChange({ showSatellites: !settings.showSatellites })}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.showSatellites ? 'bg-yellow-600' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.showSatellites ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>

            <label className="flex items-center justify-between cursor-pointer group">
              <span className="text-gray-300 group-hover:text-white transition-colors">
                Auto Rotate
              </span>
              <button
                onClick={() => onSettingsChange({ autoRotate: !settings.autoRotate })}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.autoRotate ? 'bg-purple-600' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.autoRotate ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>
          </div>
        </div>

        {/* Theme and Notifications */}
        <div className="bg-black/30 rounded-lg p-4 border border-white/5">
          <h3 className="text-white font-medium mb-4">System Settings</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Theme</span>
              <button
                onClick={() => onSettingsChange({ 
                  theme: settings.theme === 'dark' ? 'light' : 'dark' 
                })}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/40 border border-white/20 text-white hover:bg-black/60 transition-colors"
              >
                {settings.theme === 'dark' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
                {settings.theme === 'dark' ? 'Dark' : 'Light'}
              </button>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Units</span>
              <select
                value={settings.units}
                onChange={(e) => onSettingsChange({ units: e.target.value as 'metric' | 'imperial' })}
                className="bg-black/40 border border-white/20 rounded px-2 py-1 text-white text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="metric">Metric</option>
                <option value="imperial">Imperial</option>
              </select>
            </div>

            <label className="flex items-center justify-between cursor-pointer group">
              <div className="flex items-center gap-2">
                {settings.notifications ? <Bell className="w-4 h-4 text-blue-400" /> : <BellOff className="w-4 h-4 text-gray-500" />}
                <span className="text-gray-300 group-hover:text-white transition-colors">
                  Notifications
                </span>
              </div>
              <button
                onClick={() => onSettingsChange({ notifications: !settings.notifications })}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings.notifications ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.notifications ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>
          </div>
        </div>

        {/* Alert Threshold */}
        <div className="bg-black/30 rounded-lg p-4 border border-white/5">
          <h3 className="text-white font-medium mb-4 flex items-center gap-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            Alert Settings
          </h3>
          
          <div className="space-y-3">
            <div>
              <label className="block text-gray-300 text-sm mb-2">
                Alert Threshold: {settings.alertThreshold} km
              </label>
              <input
                type="range"
                min="1"
                max="100"
                value={settings.alertThreshold}
                onChange={(e) => onSettingsChange({ alertThreshold: Number(e.target.value) })}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1 km</span>
                <span>100 km</span>
              </div>
            </div>
            
            <div className="text-xs text-gray-400 bg-black/20 rounded p-2 border border-white/5">
              Objects closer than {settings.alertThreshold} km will trigger collision alerts
            </div>
          </div>
        </div>

        {/* Debug Mode */}
        {onToggleDebug && (
          <div className="bg-black/30 rounded-lg p-4 border border-white/5">
            <h3 className="text-white font-medium mb-4">Developer Options</h3>
            
            <label className="flex items-center justify-between cursor-pointer group">
              <span className="text-gray-300 group-hover:text-white transition-colors">
                Debug Mode
              </span>
              <button
                onClick={onToggleDebug}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  debugMode ? 'bg-orange-600' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                  debugMode ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>
            
            {debugMode && (
              <div className="mt-3 text-xs text-orange-400 bg-orange-900/20 rounded p-2 border border-orange-800/30">
                Debug mode enabled - Additional performance metrics and logs are active
              </div>
            )}
          </div>
        )}

        {/* Performance Info */}
        <div className="bg-black/30 rounded-lg p-4 border border-white/5">
          <h3 className="text-white font-medium mb-4">System Status</h3>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">WebGL Status:</span>
              <span className="text-green-400 font-mono">Active</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Frame Rate:</span>
              <span className="text-green-400 font-mono">60 FPS</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Memory Usage:</span>
              <span className="text-yellow-400 font-mono">~45MB</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Controls;