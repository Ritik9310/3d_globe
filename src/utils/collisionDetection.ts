import { DebrisObject, SpacecraftObject, CollisionAlert } from '../types';
import { OrbitCalculator } from './orbits';

export class CollisionDetectionSystem {
  private static readonly EARTH_RADIUS = 6371; // km
  private static readonly CRITICAL_DISTANCE = 5; // km
  private static readonly WARNING_DISTANCE = 25; // km
  private static readonly PREDICTION_HOURS = 168; // 7 days

  /**
   * Calculate 3D distance between two objects
   */
  static calculateDistance(
    obj1: { x: number; y: number; z: number },
    obj2: { x: number; y: number; z: number }
  ): number {
    return Math.sqrt(
      Math.pow(obj2.x - obj1.x, 2) +
      Math.pow(obj2.y - obj1.y, 2) +
      Math.pow(obj2.z - obj1.z, 2)
    );
  }

  /**
   * Check for immediate collision risks
   */
  static detectImmediateThreats(
    debris: DebrisObject[],
    spacecraft: SpacecraftObject[],
    threshold: number = 25
  ): CollisionAlert[] {
    const alerts: CollisionAlert[] = [];

    spacecraft.forEach(craft => {
      debris.forEach(debrisObj => {
        const distance = this.calculateDistance(craft.position, debrisObj.position);
        
        if (distance <= threshold) {
          const riskLevel = this.assessRiskLevel(distance, debrisObj.size);
          const probability = this.calculateCollisionProbability(
            distance,
            craft.velocity,
            debrisObj.velocity,
            debrisObj.size
          );

          alerts.push({
            id: `ALERT-${craft.id}-${debrisObj.id}-${Date.now()}`,
            debrisId: debrisObj.id,
            targetId: craft.id,
            riskLevel,
            estimatedDistance: distance,
            timeToClosestApproach: this.calculateTimeToClosestApproach(
              craft.position,
              craft.velocity,
              debrisObj.position,
              debrisObj.velocity
            ),
            probability,
            suggestedManeuver: this.calculateAvoidanceManeuver(
              craft.position,
              craft.velocity,
              debrisObj.position,
              debrisObj.velocity,
              distance
            ),
            createdAt: new Date()
          });
        }
      });
    });

    return alerts;
  }

  /**
   * Predict future collision risks using orbital mechanics
   */
  static predictFutureCollisions(
    debris: DebrisObject[],
    spacecraft: SpacecraftObject[],
    hoursAhead: number = 24
  ): CollisionAlert[] {
    const futureAlerts: CollisionAlert[] = [];

    spacecraft.forEach(craft => {
      debris.forEach(debrisObj => {
        if (!debrisObj.tle) return;

        try {
          // Generate future positions for both objects
          const debrisOrbit = OrbitCalculator.generateOrbitPositions(debrisObj, hoursAhead, 100);
          const craftFuturePositions = this.generateCraftFuturePositions(craft, hoursAhead, 100);

          // Check for close approaches
          const closeApproaches = this.findCloseApproaches(
            craftFuturePositions,
            debrisOrbit.positions,
            this.WARNING_DISTANCE
          );

          closeApproaches.forEach(approach => {
            const riskLevel = this.assessRiskLevel(approach.distance, debrisObj.size);
            
            futureAlerts.push({
              id: `FUTURE-ALERT-${craft.id}-${debrisObj.id}-${approach.time}`,
              debrisId: debrisObj.id,
              targetId: craft.id,
              riskLevel,
              estimatedDistance: approach.distance,
              timeToClosestApproach: approach.time / 3600, // Convert to hours
              probability: this.calculateFutureProbability(approach.distance, debrisObj.size),
              suggestedManeuver: this.calculatePreventiveManeuver(
                approach.craftPosition,
                approach.debrisPosition,
                approach.time
              ),
              createdAt: new Date()
            });
          });

        } catch (error) {
          console.warn(`Failed to predict collisions for ${debrisObj.id}:`, error);
        }
      });
    });

    return futureAlerts;
  }

  /**
   * Assess risk level based on distance and debris size
   */
  private static assessRiskLevel(distance: number, debrisSize: number): 'low' | 'medium' | 'high' | 'critical' {
    const sizeMultiplier = Math.log10(debrisSize * 100 + 1);
    
    if (distance < this.CRITICAL_DISTANCE * sizeMultiplier) return 'critical';
    if (distance < this.CRITICAL_DISTANCE * 2 * sizeMultiplier) return 'high';
    if (distance < this.WARNING_DISTANCE * sizeMultiplier) return 'medium';
    return 'low';
  }

  /**
   * Calculate collision probability using statistical methods
   */
  private static calculateCollisionProbability(
    distance: number,
    craftVelocity: { x: number; y: number; z: number },
    debrisVelocity: { x: number; y: number; z: number },
    debrisSize: number
  ): number {
    // Relative velocity
    const relativeVelocity = Math.sqrt(
      Math.pow(craftVelocity.x - debrisVelocity.x, 2) +
      Math.pow(craftVelocity.y - debrisVelocity.y, 2) +
      Math.pow(craftVelocity.z - debrisVelocity.z, 2)
    );

    // Cross-sectional area of debris
    const crossSection = Math.PI * Math.pow(debrisSize / 2, 2);
    
    // Simplified probability calculation
    const baseProbability = Math.exp(-distance / this.CRITICAL_DISTANCE);
    const velocityFactor = Math.min(relativeVelocity / 15, 2); // Normalize to orbital velocities
    const sizeFactor = Math.min(crossSection * 10000, 10); // Size importance
    
    return Math.min(baseProbability * velocityFactor * sizeFactor, 1.0);
  }

  /**
   * Calculate time to closest approach
   */
  private static calculateTimeToClosestApproach(
    pos1: { x: number; y: number; z: number },
    vel1: { x: number; y: number; z: number },
    pos2: { x: number; y: number; z: number },
    vel2: { x: number; y: number; z: number }
  ): number {
    // Relative position and velocity
    const relPos = {
      x: pos2.x - pos1.x,
      y: pos2.y - pos1.y,
      z: pos2.z - pos1.z
    };
    
    const relVel = {
      x: vel2.x - vel1.x,
      y: vel2.y - vel1.y,
      z: vel2.z - vel1.z
    };

    // Time to closest approach (in seconds)
    const relSpeed2 = relVel.x * relVel.x + relVel.y * relVel.y + relVel.z * relVel.z;
    
    if (relSpeed2 < 1e-10) return Infinity; // Objects moving together
    
    const timeToCA = -(relPos.x * relVel.x + relPos.y * relVel.y + relPos.z * relVel.z) / relSpeed2;
    
    return Math.max(0, timeToCA / 3600); // Convert to hours, ensure non-negative
  }

  /**
   * Calculate avoidance maneuver
   */
  private static calculateAvoidanceManeuver(
    craftPos: { x: number; y: number; z: number },
    craftVel: { x: number; y: number; z: number },
    debrisPos: { x: number; y: number; z: number },
    debrisVel: { x: number; y: number; z: number },
    currentDistance: number
  ): { deltaV: number; direction: string; executionTime: Date } | undefined {
    
    if (currentDistance > this.WARNING_DISTANCE) return undefined;

    // Calculate required velocity change for avoidance
    const safeDistance = Math.max(this.WARNING_DISTANCE, currentDistance * 2);
    const relativeVel = Math.sqrt(
      Math.pow(craftVel.x - debrisVel.x, 2) +
      Math.pow(craftVel.y - debrisVel.y, 2) +
      Math.pow(craftVel.z - debrisVel.z, 2)
    );

    // Simple avoidance calculation
    const requiredDeltaV = (safeDistance - currentDistance) / 100; // Simplified
    
    // Determine best maneuver direction (perpendicular to approach vector)
    const approachVector = {
      x: debrisPos.x - craftPos.x,
      y: debrisPos.y - craftPos.y,
      z: debrisPos.z - craftPos.z
    };
    
    const mag = Math.sqrt(approachVector.x ** 2 + approachVector.y ** 2 + approachVector.z ** 2);
    let direction = 'radial-out';
    
    if (approachVector.z / mag > 0.7) direction = 'anti-zenith';
    else if (approachVector.z / mag < -0.7) direction = 'zenith';
    else direction = Math.abs(approachVector.x) > Math.abs(approachVector.y) ? 'cross-track' : 'along-track';

    return {
      deltaV: Math.abs(requiredDeltaV),
      direction,
      executionTime: new Date(Date.now() + 30 * 60 * 1000) // 30 minutes from now
    };
  }

  /**
   * Generate future positions for spacecraft (simplified orbital mechanics)
   */
  private static generateCraftFuturePositions(
    craft: SpacecraftObject,
    hours: number,
    steps: number
  ): Array<{ x: number; y: number; z: number; time: number }> {
    const positions = [];
    const timeStep = (hours * 3600) / steps;
    
    // Simplified circular orbit assumption
    const r = Math.sqrt(craft.position.x ** 2 + craft.position.y ** 2 + craft.position.z ** 2);
    const v = Math.sqrt(craft.velocity.x ** 2 + craft.velocity.y ** 2 + craft.velocity.z ** 2);
    const omega = v / r; // Angular velocity
    
    for (let i = 0; i <= steps; i++) {
      const time = i * timeStep;
      const angle = omega * time;
      
      positions.push({
        x: craft.position.x * Math.cos(angle) - craft.position.y * Math.sin(angle),
        y: craft.position.x * Math.sin(angle) + craft.position.y * Math.cos(angle),
        z: craft.position.z,
        time
      });
    }
    
    return positions;
  }

  /**
   * Find close approaches between two orbital tracks
   */
  private static findCloseApproaches(
    craftPositions: Array<{ x: number; y: number; z: number; time: number }>,
    debrisPositions: Array<{ x: number; y: number; z: number; time: number }>,
    threshold: number
  ): Array<{
    distance: number;
    time: number;
    craftPosition: { x: number; y: number; z: number };
    debrisPosition: { x: number; y: number; z: number };
  }> {
    const closeApproaches = [];
    
    // Simple nearest-neighbor search
    craftPositions.forEach(craftPos => {
      debrisPositions.forEach(debrisPos => {
        if (Math.abs(craftPos.time - debrisPos.time) < 3600) { // Within 1 hour
          const distance = this.calculateDistance(craftPos, debrisPos);
          
          if (distance <= threshold) {
            closeApproaches.push({
              distance,
              time: craftPos.time,
              craftPosition: craftPos,
              debrisPosition: debrisPos
            });
          }
        }
      });
    });
    
    return closeApproaches;
  }

  /**
   * Calculate future collision probability
   */
  private static calculateFutureProbability(distance: number, debrisSize: number): number {
    const sizeFactor = Math.min(debrisSize * 10, 1);
    const distanceFactor = Math.exp(-distance / this.WARNING_DISTANCE);
    return Math.min(sizeFactor * distanceFactor, 1.0);
  }

  /**
   * Calculate preventive maneuver for future collision
   */
  private static calculatePreventiveManeuver(
    craftPos: { x: number; y: number; z: number },
    debrisPos: { x: number; y: number; z: number },
    timeToEvent: number
  ): { deltaV: number; direction: string; executionTime: Date } {
    const distance = this.calculateDistance(craftPos, debrisPos);
    const deltaV = Math.max(0.1, distance / 1000); // Minimum 0.1 m/s
    
    return {
      deltaV,
      direction: 'optimal-avoidance',
      executionTime: new Date(Date.now() + (timeToEvent - 3600) * 1000) // 1 hour before event
    };
  }

  /**
   * Advanced conjunction analysis with uncertainty propagation
   */
  static performConjunctionAnalysis(
    debris: DebrisObject,
    spacecraft: SpacecraftObject,
    timeWindow: number = 24
  ): {
    probability: number;
    missDistance: number;
    timeOfClosestApproach: Date;
    uncertainty: {
      position: number;
      velocity: number;
    };
  } | null {
    try {
      if (!debris.tle) return null;

      // Generate high-resolution orbits
      const debrisOrbit = OrbitCalculator.generateOrbitPositions(debris, timeWindow, 1000);
      const craftPositions = this.generateCraftFuturePositions(spacecraft, timeWindow, 1000);

      // Find minimum distance
      let minDistance = Infinity;
      let closestApproachTime = new Date();
      let closestCraftPos = craftPositions[0];
      let closestDebrisPos = debrisOrbit.positions[0];

      craftPositions.forEach(craftPos => {
        const timeIndex = Math.floor((craftPos.time / (timeWindow * 3600)) * debrisOrbit.positions.length);
        if (timeIndex < debrisOrbit.positions.length) {
          const debrisPos = debrisOrbit.positions[timeIndex];
          const distance = this.calculateDistance(craftPos, debrisPos);
          
          if (distance < minDistance) {
            minDistance = distance;
            closestApproachTime = new Date(Date.now() + craftPos.time * 1000);
            closestCraftPos = craftPos;
            closestDebrisPos = debrisPos;
          }
        }
      });

      // Calculate uncertainties (simplified)
      const positionUncertainty = Math.min(0.1 + debris.altitude / 10000, 5.0); // km
      const velocityUncertainty = 0.01 + debris.size * 0.001; // km/s

      // Advanced probability calculation including uncertainties
      const crossSection = Math.PI * Math.pow(debris.size / 2, 2);
      const hardBodyRadius = Math.sqrt(crossSection / Math.PI);
      const totalRadius = hardBodyRadius + positionUncertainty;
      
      const collisionProbability = Math.exp(-(minDistance ** 2) / (2 * totalRadius ** 2));

      return {
        probability: Math.min(collisionProbability, 1.0),
        missDistance: minDistance,
        timeOfClosestApproach: closestApproachTime,
        uncertainty: {
          position: positionUncertainty,
          velocity: velocityUncertainty
        }
      };

    } catch (error) {
      console.warn('Conjunction analysis failed:', error);
      return null;
    }
  }
}

export default CollisionDetectionSystem;