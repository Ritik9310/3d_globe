import React, { useState, useEffect } from 'react';
import { CollisionAlert } from '../types';
import { AlertTriangle, X, Clock, Navigation, Zap, Info, CheckCircle } from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';

interface AlertsProps {
  alerts: CollisionAlert[];
  onDismissAlert: (alertId: string) => void;
  onExecuteManeuver: (alert: CollisionAlert) => void;
}

const Alerts: React.FC<AlertsProps> = ({ alerts, onDismissAlert, onExecuteManeuver }) => {
  const [expandedAlert, setExpandedAlert] = useState<string | null>(null);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  // Filter out dismissed alerts
  const activeAlerts = alerts.filter(alert => !dismissedAlerts.has(alert.id));

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return { bg: 'bg-red-900/20', border: 'border-red-500/30', text: 'text-red-400', accent: 'bg-red-500' };
      case 'high': return { bg: 'bg-orange-900/20', border: 'border-orange-500/30', text: 'text-orange-400', accent: 'bg-orange-500' };
      case 'medium': return { bg: 'bg-yellow-900/20', border: 'border-yellow-500/30', text: 'text-yellow-400', accent: 'bg-yellow-500' };
      case 'low': return { bg: 'bg-blue-900/20', border: 'border-blue-500/30', text: 'text-blue-400', accent: 'bg-blue-500' };
      default: return { bg: 'bg-gray-900/20', border: 'border-gray-500/30', text: 'text-gray-400', accent: 'bg-gray-500' };
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return <AlertTriangle className="w-5 h-5 animate-pulse" />;
      case 'high': return <AlertTriangle className="w-5 h-5" />;
      case 'medium': return <Clock className="w-5 h-5" />;
      case 'low': return <Info className="w-5 h-5" />;
      default: return <Info className="w-5 h-5" />;
    }
  };

  const handleDismiss = (alertId: string) => {
    setDismissedAlerts(prev => new Set([...prev, alertId]));
    onDismissAlert(alertId);
  };

  const handleExecuteManeuver = (alert: CollisionAlert) => {
    onExecuteManeuver(alert);
    // Show success feedback
    setTimeout(() => {
      handleDismiss(alert.id);
    }, 2000);
  };

  // Auto-dismiss low-risk alerts after 30 seconds
  useEffect(() => {
    const timer = setInterval(() => {
      activeAlerts.forEach(alert => {
        if (alert.riskLevel === 'low' && 
            Date.now() - alert.createdAt.getTime() > 30000) {
          handleDismiss(alert.id);
        }
      });
    }, 5000);

    return () => clearInterval(timer);
  }, [activeAlerts]);

  if (activeAlerts.length === 0) {
    return (
      <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <h2 className="text-xl font-semibold text-white">All Clear</h2>
        </div>
        <div className="text-gray-400 text-center py-8">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-900/20 flex items-center justify-center">
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
          <p className="text-lg mb-2">No Active Collision Alerts</p>
          <p className="text-sm text-gray-500">Space traffic is operating within safe parameters</p>
        </div>
      </div>
    );
  }

  // Sort alerts by risk level and time
  const sortedAlerts = [...activeAlerts].sort((a, b) => {
    const riskPriority = { critical: 4, high: 3, medium: 2, low: 1 };
    if (riskPriority[a.riskLevel] !== riskPriority[b.riskLevel]) {
      return riskPriority[b.riskLevel] - riskPriority[a.riskLevel];
    }
    return a.timeToClosestApproach - b.timeToClosestApproach;
  });

  return (
    <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="relative">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            {sortedAlerts.some(a => a.riskLevel === 'critical' || a.riskLevel === 'high') && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            )}
          </div>
          <h2 className="text-xl font-semibold text-white">Collision Alerts</h2>
          <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded-full border border-red-500/30">
            {sortedAlerts.length} Active
          </span>
        </div>
      </div>

      <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
        {sortedAlerts.map((alert) => {
          const colors = getRiskColor(alert.riskLevel);
          const isExpanded = expandedAlert === alert.id;
          const timeToApproach = alert.timeToClosestApproach;

          return (
            <div
              key={alert.id}
              className={`${colors.bg} ${colors.border} border rounded-lg transition-all duration-200 ${
                isExpanded ? 'shadow-lg' : 'hover:shadow-md'
              }`}
            >
              <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-start gap-3">
                    <div className={`${colors.text} mt-1`}>
                      {getRiskIcon(alert.riskLevel)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-2 py-1 text-xs font-semibold rounded-full ${colors.accent} text-white`}>
                          {alert.riskLevel.toUpperCase()}
                        </span>
                        <span className="text-gray-400 text-sm font-mono">
                          {alert.debrisId} → {alert.targetId}
                        </span>
                      </div>
                      <div className="space-y-1 text-sm">
                        <div className="flex items-center gap-4">
                          <span className="text-gray-400">Distance:</span>
                          <span className={`font-mono ${colors.text}`}>
                            {alert.estimatedDistance.toFixed(1)} km
                          </span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-gray-400">Time to CA:</span>
                          <span className={`font-mono ${colors.text}`}>
                            {timeToApproach < 1 
                              ? `${Math.round(timeToApproach * 60)}m`
                              : `${timeToApproach.toFixed(1)}h`
                            }
                          </span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-gray-400">Probability:</span>
                          <span className={`font-mono ${colors.text}`}>
                            {(alert.probability * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={() => setExpandedAlert(isExpanded ? null : alert.id)}
                      className={`px-3 py-1.5 text-xs rounded border transition-colors ${colors.border} ${colors.text} hover:bg-white/5`}
                    >
                      {isExpanded ? 'Collapse' : 'Details'}
                    </button>
                    <button
                      onClick={() => handleDismiss(alert.id)}
                      className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-white/10 pt-4 mt-4 space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400 block">Created:</span>
                        <span className="text-white font-mono">
                          {format(alert.createdAt, 'MMM dd, HH:mm:ss')}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400 block">Age:</span>
                        <span className="text-white font-mono">
                          {formatDistanceToNow(alert.createdAt)} ago
                        </span>
                      </div>
                    </div>

                    {alert.suggestedManeuver && (
                      <div className="bg-black/30 rounded-lg p-4 border border-white/10">
                        <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                          <Navigation className="w-4 h-4" />
                          Suggested Maneuver
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Delta-V:</span>
                            <span className="text-white font-mono">
                              {alert.suggestedManeuver.deltaV.toFixed(3)} m/s
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Direction:</span>
                            <span className="text-white font-mono">
                              {alert.suggestedManeuver.direction}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Execute at:</span>
                            <span className="text-white font-mono">
                              {format(alert.suggestedManeuver.executionTime, 'HH:mm:ss')}
                            </span>
                          </div>
                        </div>
                        
                        <div className="flex gap-2 mt-4">
                          <button
                            onClick={() => handleExecuteManeuver(alert)}
                            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
                          >
                            <Zap className="w-4 h-4" />
                            Execute Maneuver
                          </button>
                          <button
                            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors text-sm"
                          >
                            Simulate
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Progress bar for time to closest approach */}
                <div className="mt-3">
                  <div className="h-1 bg-black/30 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${colors.accent} transition-all duration-1000`}
                      style={{
                        width: `${Math.max(5, Math.min(95, (24 - timeToApproach) / 24 * 100))}%`
                      }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Now</span>
                    <span>24h</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Statistics */}
      <div className="mt-6 pt-4 border-t border-white/10">
        <div className="grid grid-cols-4 gap-4 text-center">
          {['critical', 'high', 'medium', 'low'].map(level => {
            const count = sortedAlerts.filter(a => a.riskLevel === level).length;
            const colors = getRiskColor(level);
            return (
              <div key={level}>
                <div className={`text-2xl font-bold ${colors.text}`}>{count}</div>
                <div className="text-xs text-gray-400 capitalize">{level}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Alerts;