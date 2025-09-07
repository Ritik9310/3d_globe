import React, { useEffect, useRef, useState } from 'react';
import { DebrisObject, SpacecraftObject, SystemSettings } from '../types';
import { OrbitCalculator } from '../utils/orbits';

interface GlobeProps {
  debris: DebrisObject[];
  spacecraft: SpacecraftObject[];
  settings: SystemSettings;
  selectedObject?: DebrisObject | SpacecraftObject;
  onObjectSelect: (object: DebrisObject | SpacecraftObject) => void;
}

// Mock Cesium-like API for demonstration
class MockCesium {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private camera = { position: { x: 0, y: 0, z: 15000 }, rotation: 0 };
  private animationFrame: number | null = null;
  private objects: Array<{ position: { x: number; y: number; z: number }, color: string, size: number, id: string }> = [];
  private orbits: Array<{ positions: Array<{ x: number; y: number; z: number }>, color: string }> = [];

  constructor(container: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = container.clientWidth;
    this.canvas.height = container.clientHeight;
    this.canvas.style.background = 'linear-gradient(to bottom, #000428, #004e92)';
    container.appendChild(this.canvas);

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    this.ctx = ctx;

    this.render();
  }

  addDebris(debris: DebrisObject[]) {
    this.objects = debris.map(d => ({
      position: d.position,
      color: this.getRiskColor(d.riskLevel),
      size: Math.max(2, Math.log(d.size * 100 + 1) * 2),
      id: d.id
    }));
  }

  addSpacecraft(spacecraft: SpacecraftObject[]) {
    spacecraft.forEach(s => {
      this.objects.push({
        position: s.position,
        color: s.type === 'ISS' ? '#00ff00' : '#ffff00',
        size: 8,
        id: s.id
      });
    });
  }

  addOrbit(debris: DebrisObject, show: boolean) {
    if (!show || !debris.tle) return;

    try {
      const orbitData = OrbitCalculator.generateOrbitPositions(debris, 2, 50);
      this.orbits.push({
        positions: orbitData.positions,
        color: this.getRiskColor(debris.riskLevel)
      });
    } catch (error) {
      console.warn('Failed to generate orbit for', debris.id, error);
    }
  }

  private getRiskColor(riskLevel: string): string {
    switch (riskLevel) {
      case 'critical': return '#ff0000';
      case 'high': return '#ff6600';
      case 'medium': return '#ffcc00';
      case 'low': return '#00ff00';
      default: return '#ffffff';
    }
  }

  private project3D(pos: { x: number; y: number; z: number }): { x: number; y: number, visible: boolean } {
    // Simple orthographic projection with rotation
    const cos = Math.cos(this.camera.rotation);
    const sin = Math.sin(this.camera.rotation);
    
    const rotX = pos.x * cos - pos.y * sin;
    const rotY = pos.x * sin + pos.y * cos;
    
    const scale = 6371 / (Math.abs(pos.z - this.camera.position.z) + 6371) * 50;
    
    return {
      x: this.canvas.width / 2 + rotX * scale,
      y: this.canvas.height / 2 + rotY * scale,
      visible: pos.z - this.camera.position.z > -20000
    };
  }

  private drawStars() {
    this.ctx.fillStyle = '#ffffff';
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * this.canvas.width;
      const y = Math.random() * this.canvas.height;
      const size = Math.random() * 2;
      this.ctx.globalAlpha = Math.random() * 0.8 + 0.2;
      this.ctx.fillRect(x, y, size, size);
    }
    this.ctx.globalAlpha = 1;
  }

  private drawEarth() {
    const center = this.project3D({ x: 0, y: 0, z: 0 });
    const radius = 6371 * 50 / (Math.abs(-this.camera.position.z) + 6371);
    
    // Earth body
    const gradient = this.ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, radius);
    gradient.addColorStop(0, '#4a90e2');
    gradient.addColorStop(0.7, '#357abd');
    gradient.addColorStop(1, '#1e3a5f');
    
    this.ctx.fillStyle = gradient;
    this.ctx.beginPath();
    this.ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI);
    this.ctx.fill();

    // Continents (simplified)
    this.ctx.fillStyle = '#2d5a2d';
    this.ctx.globalAlpha = 0.6;
    for (let i = 0; i < 20; i++) {
      const angle = (i * 18) * Math.PI / 180;
      const x = center.x + Math.cos(angle) * radius * 0.7 * Math.random();
      const y = center.y + Math.sin(angle) * radius * 0.7 * Math.random();
      const size = radius * 0.1 * Math.random();
      this.ctx.beginPath();
      this.ctx.arc(x, y, size, 0, 2 * Math.PI);
      this.ctx.fill();
    }
    this.ctx.globalAlpha = 1;
  }

  private render = () => {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Stars background
    this.drawStars();
    
    // Earth
    this.drawEarth();
    
    // Orbits
    this.orbits.forEach(orbit => {
      this.ctx.strokeStyle = orbit.color;
      this.ctx.globalAlpha = 0.3;
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      
      orbit.positions.forEach((pos, i) => {
        const projected = this.project3D(pos);
        if (projected.visible) {
          if (i === 0) {
            this.ctx.moveTo(projected.x, projected.y);
          } else {
            this.ctx.lineTo(projected.x, projected.y);
          }
        }
      });
      
      this.ctx.stroke();
      this.ctx.globalAlpha = 1;
    });
    
    // Objects
    this.objects.forEach(obj => {
      const projected = this.project3D(obj.position);
      if (projected.visible) {
        this.ctx.fillStyle = obj.color;
        this.ctx.beginPath();
        this.ctx.arc(projected.x, projected.y, obj.size, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Add glow effect
        this.ctx.globalAlpha = 0.5;
        this.ctx.beginPath();
        this.ctx.arc(projected.x, projected.y, obj.size + 2, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.globalAlpha = 1;
      }
    });
    
    // Auto-rotate
    this.camera.rotation += 0.005;
    
    this.animationFrame = requestAnimationFrame(this.render);
  };

  destroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.canvas.remove();
  }

  zoomToISS() {
    this.camera.position.z = 8000;
  }

  zoomToEarth() {
    this.camera.position.z = 15000;
  }

  setAutoRotate(enabled: boolean) {
    // Auto-rotate is always enabled in this demo
  }
}

const Globe: React.FC<GlobeProps> = ({
  debris,
  spacecraft,
  settings,
  selectedObject,
  onObjectSelect
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cesiumRef = useRef<MockCesium | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Cesium globe
    try {
      cesiumRef.current = new MockCesium(containerRef.current);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to initialize globe:', error);
    }

    return () => {
      if (cesiumRef.current) {
        cesiumRef.current.destroy();
      }
    };
  }, []);

  useEffect(() => {
    if (!cesiumRef.current) return;

    // Update debris visualization
    cesiumRef.current.addDebris(debris);
    cesiumRef.current.addSpacecraft(spacecraft);

    // Add orbits if enabled
    if (settings.showOrbits) {
      debris.slice(0, 50).forEach(d => {
        cesiumRef.current?.addOrbit(d, true);
      });
    }
  }, [debris, spacecraft, settings.showOrbits]);

  const handleZoomToISS = () => {
    cesiumRef.current?.zoomToISS();
  };

  const handleZoomToEarth = () => {
    cesiumRef.current?.zoomToEarth();
  };

  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gradient-to-b from-gray-900 to-black">
        <div className="text-white text-center">
          <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p>Loading 3D Globe...</p>
          <p className="text-sm text-gray-400 mt-2">Initializing Cesium visualization</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full overflow-hidden bg-black">
      {/* Globe Container */}
      <div 
        ref={containerRef} 
        className="w-full h-full"
        style={{ cursor: 'grab' }}
      />
      
      {/* Control Overlay */}
      <div className="absolute top-4 left-4 bg-black/30 backdrop-blur-sm rounded-lg p-4 border border-white/10">
        <div className="flex flex-col gap-2">
          <button
            onClick={handleZoomToISS}
            className="px-3 py-2 bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded border border-green-500/30 transition-colors"
          >
            Zoom to ISS
          </button>
          <button
            onClick={handleZoomToEarth}
            className="px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded border border-blue-500/30 transition-colors"
          >
            View Earth
          </button>
        </div>
      </div>

      {/* Statistics Overlay */}
      <div className="absolute top-4 right-4 bg-black/30 backdrop-blur-sm rounded-lg p-4 border border-white/10">
        <div className="text-white text-sm space-y-2">
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Debris Objects:</span>
            <span className="text-white font-mono">{debris.length.toLocaleString()}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">Spacecraft:</span>
            <span className="text-white font-mono">{spacecraft.length}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-gray-400">High Risk:</span>
            <span className="text-red-400 font-mono">
              {debris.filter(d => d.riskLevel === 'high' || d.riskLevel === 'critical').length}
            </span>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-black/30 backdrop-blur-sm rounded-lg p-4 border border-white/10">
        <h3 className="text-white font-semibold mb-3">Risk Levels</h3>
        <div className="space-y-2">
          {[
            { level: 'Critical', color: '#ff0000' },
            { level: 'High', color: '#ff6600' },
            { level: 'Medium', color: '#ffcc00' },
            { level: 'Low', color: '#00ff00' }
          ].map(({ level, color }) => (
            <div key={level} className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-gray-300 text-sm">{level}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 mt-3 pt-2 border-t border-white/10">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-gray-300 text-sm">ISS</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <span className="text-gray-300 text-sm">Satellites</span>
          </div>
        </div>
      </div>

      {/* Loading indicator for data updates */}
      {debris.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="text-white text-center">
            <div className="animate-pulse w-8 h-8 bg-blue-500 rounded-full mx-auto mb-2"></div>
            <p>Loading debris data...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Globe;