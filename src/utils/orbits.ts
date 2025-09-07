import { DebrisObject, OrbitData } from '../types';

// Earth constants
const EARTH_RADIUS = 6371; // km
const MU = 398600.4418; // km³/s² - Earth's gravitational parameter
const J2 = 1.08262668e-3; // Earth's oblateness coefficient

export class OrbitCalculator {
  /**
   * Parse Two-Line Element (TLE) data into orbital parameters
   */
  static parseTLE(line1: string, line2: string): {
    inclination: number;
    raan: number;
    eccentricity: number;
    argOfPerigee: number;
    meanAnomaly: number;
    meanMotion: number;
    epoch: Date;
  } {
    // Parse TLE format
    const inclination = parseFloat(line2.substring(8, 16));
    const raan = parseFloat(line2.substring(17, 25)); // Right Ascension of Ascending Node
    const eccentricity = parseFloat('0.' + line2.substring(26, 33));
    const argOfPerigee = parseFloat(line2.substring(34, 42));
    const meanAnomaly = parseFloat(line2.substring(43, 51));
    const meanMotion = parseFloat(line2.substring(52, 63)); // revolutions per day

    // Parse epoch from line 1
    const epochYear = parseInt(line1.substring(18, 20));
    const epochDay = parseFloat(line1.substring(20, 32));
    const year = epochYear < 57 ? 2000 + epochYear : 1900 + epochYear;
    const epoch = new Date(year, 0, epochDay);

    return {
      inclination,
      raan,
      eccentricity,
      argOfPerigee,
      meanAnomaly,
      meanMotion,
      epoch,
    };
  }

  /**
   * Convert orbital elements to Cartesian coordinates
   */
  static orbitalToCartesian(
    semiMajorAxis: number,
    eccentricity: number,
    inclination: number,
    raan: number,
    argOfPerigee: number,
    trueAnomaly: number
  ): { x: number; y: number; z: number } {
    const incRad = (inclination * Math.PI) / 180;
    const raanRad = (raan * Math.PI) / 180;
    const argPerRad = (argOfPerigee * Math.PI) / 180;

    // Position in orbital plane
    const r = (semiMajorAxis * (1 - eccentricity * eccentricity)) / (1 + eccentricity * Math.cos(trueAnomaly));
    const xOrbital = r * Math.cos(trueAnomaly);
    const yOrbital = r * Math.sin(trueAnomaly);

    // Rotation matrices
    const cosRaan = Math.cos(raanRad);
    const sinRaan = Math.sin(raanRad);
    const cosInc = Math.cos(incRad);
    const sinInc = Math.sin(incRad);
    const cosArgPer = Math.cos(argPerRad);
    const sinArgPer = Math.sin(argPerRad);

    // Transform to Earth-Centered Inertial coordinates
    const x = (cosRaan * cosArgPer - sinRaan * sinArgPer * cosInc) * xOrbital +
              (-cosRaan * sinArgPer - sinRaan * cosArgPer * cosInc) * yOrbital;
    
    const y = (sinRaan * cosArgPer + cosRaan * sinArgPer * cosInc) * xOrbital +
              (-sinRaan * sinArgPer + cosRaan * cosArgPer * cosInc) * yOrbital;
    
    const z = (sinInc * sinArgPer) * xOrbital + (sinInc * cosArgPer) * yOrbital;

    return { x, y, z };
  }

  /**
   * Calculate orbital period
   */
  static calculateOrbitalPeriod(semiMajorAxis: number): number {
    return 2 * Math.PI * Math.sqrt((semiMajorAxis * semiMajorAxis * semiMajorAxis) / MU);
  }

  /**
   * Generate orbit positions over time
   */
  static generateOrbitPositions(debris: DebrisObject, hours: number = 24, steps: number = 100): OrbitData {
    if (!debris.tle) {
      throw new Error('TLE data required for orbit generation');
    }

    const tleData = this.parseTLE(debris.tle.line1, debris.tle.line2);
    const semiMajorAxis = Math.pow(MU / ((tleData.meanMotion * 2 * Math.PI / 86400) ** 2), 1/3);
    const period = this.calculateOrbitalPeriod(semiMajorAxis);

    const positions: Array<{ x: number; y: number; z: number; time: number }> = [];
    const timeStep = (hours * 3600) / steps;

    for (let i = 0; i <= steps; i++) {
      const time = i * timeStep;
      const meanAnomaly = (tleData.meanAnomaly + (time / period) * 360) % 360;
      const trueAnomaly = this.meanAnomalyToTrueAnomaly(meanAnomaly * Math.PI / 180, tleData.eccentricity);

      const position = this.orbitalToCartesian(
        semiMajorAxis,
        tleData.eccentricity,
        tleData.inclination,
        tleData.raan,
        tleData.argOfPerigee,
        trueAnomaly
      );

      // Apply atmospheric drag effect
      const altitudeKm = Math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2) - EARTH_RADIUS;
      const dragEffect = this.calculateAtmosphericDrag(altitudeKm, debris.size, time);
      
      positions.push({
        x: position.x - dragEffect.x,
        y: position.y - dragEffect.y,
        z: position.z - dragEffect.z,
        time,
      });
    }

    return {
      positions,
      period: period / 60, // convert to minutes
      apogee: semiMajorAxis * (1 + tleData.eccentricity) - EARTH_RADIUS,
      perigee: semiMajorAxis * (1 - tleData.eccentricity) - EARTH_RADIUS,
    };
  }

  /**
   * Convert mean anomaly to true anomaly using Newton's method
   */
  private static meanAnomalyToTrueAnomaly(meanAnomaly: number, eccentricity: number): number {
    let eccentricAnomaly = meanAnomaly;
    
    // Newton's method iteration
    for (let i = 0; i < 10; i++) {
      const delta = eccentricAnomaly - eccentricity * Math.sin(eccentricAnomaly) - meanAnomaly;
      eccentricAnomaly -= delta / (1 - eccentricity * Math.cos(eccentricAnomaly));
      if (Math.abs(delta) < 1e-12) break;
    }

    // Convert eccentric anomaly to true anomaly
    const trueAnomaly = 2 * Math.atan2(
      Math.sqrt(1 + eccentricity) * Math.sin(eccentricAnomaly / 2),
      Math.sqrt(1 - eccentricity) * Math.cos(eccentricAnomaly / 2)
    );

    return trueAnomaly;
  }

  /**
   * Calculate atmospheric drag effect on orbital decay
   */
  private static calculateAtmosphericDrag(altitude: number, objectSize: number, timeElapsed: number): { x: number; y: number; z: number } {
    if (altitude > 800) return { x: 0, y: 0, z: 0 }; // Minimal drag above 800km

    // Simplified atmospheric density model
    const rho = this.getAtmosphericDensity(altitude);
    const crossSectionalArea = Math.PI * (objectSize / 2) ** 2; // Assume spherical
    const dragCoefficient = 2.2; // Typical for space debris
    
    // Drag acceleration (very simplified)
    const dragAccel = 0.5 * rho * crossSectionalArea * dragCoefficient;
    const altitudeDecay = dragAccel * (timeElapsed ** 2) / 2;

    // Apply altitude-dependent drag in the direction opposite to velocity
    const dragFactor = Math.min(altitudeDecay / 1000, altitude * 0.001); // Prevent excessive decay
    
    return {
      x: Math.random() * dragFactor * 0.1,
      y: Math.random() * dragFactor * 0.1,
      z: -dragFactor, // Primary decay is radially inward
    };
  }

  /**
   * Get atmospheric density at given altitude
   */
  private static getAtmosphericDensity(altitude: number): number {
    // Simplified exponential atmosphere model
    if (altitude < 200) return 2.5e-11; // kg/m³
    if (altitude < 300) return 1.7e-12;
    if (altitude < 400) return 3.0e-13;
    if (altitude < 500) return 7.0e-14;
    if (altitude < 600) return 2.0e-14;
    if (altitude < 700) return 7.0e-15;
    if (altitude < 800) return 3.0e-15;
    return 1.0e-15; // Very thin above 800km
  }

  /**
   * Calculate velocity vector from position and orbital parameters
   */
  static calculateVelocity(
    position: { x: number; y: number; z: number },
    semiMajorAxis: number,
    eccentricity: number,
    inclination: number,
    raan: number,
    argOfPerigee: number,
    trueAnomaly: number
  ): { x: number; y: number; z: number } {
    const r = Math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2);
    const h = Math.sqrt(MU * semiMajorAxis * (1 - eccentricity ** 2));
    
    // Velocity in orbital plane
    const vx = -(MU / h) * Math.sin(trueAnomaly);
    const vy = (MU / h) * (eccentricity + Math.cos(trueAnomaly));
    
    // Transform to inertial frame (simplified)
    const incRad = (inclination * Math.PI) / 180;
    const raanRad = (raan * Math.PI) / 180;
    const argPerRad = (argOfPerigee * Math.PI) / 180;
    
    const cosRaan = Math.cos(raanRad);
    const sinRaan = Math.sin(raanRad);
    const cosInc = Math.cos(incRad);
    const sinInc = Math.sin(incRad);
    const cosArgPer = Math.cos(argPerRad);
    const sinArgPer = Math.sin(argPerRad);
    
    return {
      x: (cosRaan * cosArgPer - sinRaan * sinArgPer * cosInc) * vx +
         (-cosRaan * sinArgPer - sinRaan * cosArgPer * cosInc) * vy,
      y: (sinRaan * cosArgPer + cosRaan * sinArgPer * cosInc) * vx +
         (-sinRaan * sinArgPer + cosRaan * cosArgPer * cosInc) * vy,
      z: (sinInc * sinArgPer) * vx + (sinInc * cosArgPer) * vy,
    };
  }
}

export default OrbitCalculator;