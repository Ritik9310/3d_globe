import { DebrisObject, SpacecraftObject } from '../types';

// Mock TLE data for various debris objects
const MOCK_TLE_DATA = {
  // Defunct satellites
  'COSMOS-2251-DEB': {
    line1: '1 34454U 93036SX  24001.50000000  .00000000  00000-0  00000-0 0    07',
    line2: '2 34454  74.0458 280.7864 0012345 123.4567 236.8901 14.12345678123456'
  },
  'IRIDIUM-33-DEB': {
    line1: '1 25077U 97051C   24001.50000000  .00000000  00000-0  00000-0 0    08',
    line2: '2 25077  86.4008 123.7864 0002345 234.5678 125.4321 14.34567891234567'
  },
  // Space station debris
  'MIR-DEB': {
    line1: '1 16609U 86017A   24001.50000000  .00000000  00000-0  00000-0 0    09',
    line2: '2 16609  51.6461  89.5432 0003456 345.6789  14.3210 15.54123456789012'
  },
  // Upper stage debris
  'ARIANE-DEB': {
    line1: '1 23439U 94089B   24001.50000000  .00000000  00000-0  00000-0 0    01',
    line2: '2 23439  98.7654 187.6543 0001234 156.7890 203.4567 14.89012345678901'
  }
};

/**
 * Generate comprehensive mock debris dataset
 */
export class DebrisDataGenerator {
  private static debrisCounter = 1000;

  static generateMockDebris(count: number = 500): DebrisObject[] {
    const debris: DebrisObject[] = [];
    const riskLevels: Array<'low' | 'medium' | 'high' | 'critical'> = ['low', 'medium', 'high', 'critical'];
    
    for (let i = 0; i < count; i++) {
      const id = `DEBRIS-${String(this.debrisCounter++).padStart(4, '0')}`;
      const riskLevel = this.getWeightedRiskLevel();
      
      // Generate orbital parameters with realistic distributions
      const altitude = this.generateRealisticAltitude();
      const inclination = this.generateInclination();
      const eccentricity = Math.random() * 0.1; // Most debris has low eccentricity
      
      // Position calculation based on orbital parameters
      const orbitRadius = 6371 + altitude; // Earth radius + altitude
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      
      const position = {
        x: orbitRadius * Math.sin(phi) * Math.cos(theta),
        y: orbitRadius * Math.sin(phi) * Math.sin(theta),
        z: orbitRadius * Math.cos(phi)
      };

      // Velocity calculation (simplified circular orbit)
      const orbitalVelocity = Math.sqrt(398600.4418 / orbitRadius); // km/s
      const velocity = {
        x: -orbitalVelocity * Math.sin(theta) + (Math.random() - 0.5) * 0.5,
        y: orbitalVelocity * Math.cos(theta) + (Math.random() - 0.5) * 0.5,
        z: (Math.random() - 0.5) * 0.2
      };

      // Size distribution based on real debris statistics
      const size = this.generateDebrisSize();
      const mass = this.calculateMass(size);

      // Generate TLE data for this debris
      const tle = this.generateTLEData(id, altitude, inclination, eccentricity);

      debris.push({
        id,
        name: this.generateDebrisName(id),
        position,
        velocity,
        size,
        mass,
        riskLevel,
        riskScore: this.calculateRiskScore(riskLevel, size, altitude),
        altitude,
        inclination,
        eccentricity,
        lastUpdate: new Date(),
        tle,
        nextCloseApproach: Math.random() > 0.7 ? this.generateCloseApproach() : undefined
      });
    }

    return debris;
  }

  /**
   * Generate realistic altitude distribution based on actual debris data
   */
  private static generateRealisticAltitude(): number {
    const random = Math.random();
    
    if (random < 0.3) {
      // LEO debris (200-800 km) - most common
      return 200 + Math.random() * 600;
    } else if (random < 0.7) {
      // MEO debris (800-20,000 km)
      return 800 + Math.random() * 19200;
    } else {
      // GEO and high elliptical (20,000-35,786 km)
      return 20000 + Math.random() * 15786;
    }
  }

  /**
   * Generate inclination with realistic distribution
   */
  private static generateInclination(): number {
    const random = Math.random();
    
    if (random < 0.4) {
      // Sun-synchronous orbits (~98-99°)
      return 98 + Math.random() * 2;
    } else if (random < 0.6) {
      // Polar orbits (85-95°)
      return 85 + Math.random() * 10;
    } else if (random < 0.8) {
      // Medium inclination (30-60°)
      return 30 + Math.random() * 30;
    } else {
      // Low inclination (0-30°)
      return Math.random() * 30;
    }
  }

  /**
   * Generate debris size with realistic distribution
   */
  private static generateDebrisSize(): number {
    const random = Math.random();
    
    if (random < 0.7) {
      // Small debris (1cm - 10cm) - most common
      return 0.01 + Math.random() * 0.09;
    } else if (random < 0.9) {
      // Medium debris (10cm - 1m)
      return 0.1 + Math.random() * 0.9;
    } else {
      // Large debris (1m - 10m)
      return 1 + Math.random() * 9;
    }
  }

  /**
   * Calculate mass based on size and material assumptions
   */
  private static calculateMass(size: number): number {
    // Assume average density of 2800 kg/m³ (aluminum)
    const volume = (4/3) * Math.PI * Math.pow(size/2, 3);
    return volume * 2800;
  }

  /**
   * Generate weighted risk levels (more low-risk objects)
   */
  private static getWeightedRiskLevel(): 'low' | 'medium' | 'high' | 'critical' {
    const random = Math.random();
    
    if (random < 0.6) return 'low';
    if (random < 0.85) return 'medium';
    if (random < 0.97) return 'high';
    return 'critical';
  }

  /**
   * Calculate risk score based on multiple factors
   */
  private static calculateRiskScore(riskLevel: string, size: number, altitude: number): number {
    let baseScore = 0;
    
    switch (riskLevel) {
      case 'low': baseScore = 10 + Math.random() * 20; break;
      case 'medium': baseScore = 30 + Math.random() * 20; break;
      case 'high': baseScore = 50 + Math.random() * 30; break;
      case 'critical': baseScore = 80 + Math.random() * 20; break;
    }

    // Size factor (larger objects are more dangerous)
    const sizeFactor = Math.min(size * 10, 20);
    
    // Altitude factor (LEO objects are more dangerous due to higher density)
    const altitudeFactor = altitude < 800 ? 10 : altitude < 2000 ? 5 : 0;
    
    return Math.min(Math.round(baseScore + sizeFactor + altitudeFactor), 100);
  }

  /**
   * Generate realistic debris names
   */
  private static generateDebrisName(id: string): string {
    const sources = [
      'COSMOS', 'IRIDIUM', 'FENGYUN', 'CERISE', 'ENVISAT', 'SPOT',
      'ARIANE', 'DELTA', 'FALCON', 'PROTON', 'SOYUZ', 'ATLAS',
      'UNKNOWN', 'FRAGMENT', 'PAYLOAD', 'ROCKET-BODY'
    ];
    
    const suffixes = ['DEB', 'FRAG', 'PART', 'SEC', 'PIECE', 'COMP'];
    
    const source = sources[Math.floor(Math.random() * sources.length)];
    const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
    const number = Math.floor(Math.random() * 9999);
    
    return `${source}-${number}-${suffix}`;
  }

  /**
   * Generate TLE data for debris object
   */
  private static generateTLEData(id: string, altitude: number, inclination: number, eccentricity: number): { line1: string; line2: string } {
    // Simplified TLE generation
    const catalogNumber = String(Math.floor(Math.random() * 99999)).padStart(5, '0');
    const epochYear = '24';
    const epochDay = (Math.random() * 365 + 1).toFixed(8).padStart(12, '0');
    
    // Calculate mean motion from altitude
    const semiMajorAxis = 6371 + altitude;
    const meanMotion = Math.sqrt(398600.4418 / (semiMajorAxis ** 3)) * 86400 / (2 * Math.PI);
    
    const line1 = `1 ${catalogNumber}U 24001A   ${epochYear}${epochDay}  .00000000  00000-0  00000-0 0    0${Math.floor(Math.random() * 10)}`;
    const line2 = `2 ${catalogNumber} ${inclination.toFixed(4).padStart(8)} ${(Math.random() * 360).toFixed(4).padStart(8)} ${eccentricity.toFixed(7)} ${(Math.random() * 360).toFixed(4).padStart(8)} ${(Math.random() * 360).toFixed(4).padStart(8)} ${meanMotion.toFixed(8)}${String(Math.floor(Math.random() * 99999)).padStart(5, '0')}`;
    
    return { line1, line2 };
  }

  /**
   * Generate close approach data
   */
  private static generateCloseApproach() {
    return {
      targetId: 'ISS-ZARYA',
      distance: Math.random() * 50 + 5, // 5-55 km
      time: new Date(Date.now() + Math.random() * 7 * 24 * 60 * 60 * 1000) // Next 7 days
    };
  }
}

/**
 * Generate spacecraft/station data
 */
export const SPACECRAFT_DATA: SpacecraftObject[] = [
  {
    id: 'ISS-ZARYA',
    name: 'International Space Station',
    type: 'ISS',
    position: { x: -4012.45, y: 4583.12, z: 3890.67 },
    velocity: { x: -5.2341, y: -4.8932, z: 2.1234 },
    altitude: 408,
    isActive: true
  },
  {
    id: 'TIANGONG',
    name: 'Chinese Space Station',
    type: 'spacecraft',
    position: { x: 3892.34, y: -4234.56, z: 4567.89 },
    velocity: { x: 4.8932, y: 5.2341, z: -1.8765 },
    altitude: 382,
    isActive: true
  },
  {
    id: 'HUBBLE',
    name: 'Hubble Space Telescope',
    type: 'satellite',
    position: { x: 2341.67, y: 5234.89, z: -3456.78 },
    velocity: { x: -6.1234, y: 2.3456, z: 4.5678 },
    altitude: 547,
    isActive: true
  }
];

/**
 * Fetch live TLE data from external APIs
 */
export class TLEDataFetcher {
  private static readonly CELESTRAK_BASE_URL = 'https://celestrak.org/NORAD/elements/gp.php';
  private static readonly SPACE_TRACK_URL = 'https://www.space-track.org';

  /**
   * Fetch TLE data from CelesTrak
   */
  static async fetchFromCelesTrak(catalogNumber: string): Promise<{ line1: string; line2: string } | null> {
    try {
      const response = await fetch(`${this.CELESTRAK_BASE_URL}?CATNR=${catalogNumber}&FORMAT=tle`);
      if (!response.ok) return null;
      
      const tleData = await response.text();
      const lines = tleData.split('\n').filter(line => line.length > 0);
      
      if (lines.length >= 2) {
        return {
          line1: lines[lines.length - 2],
          line2: lines[lines.length - 1]
        };
      }
      
      return null;
    } catch (error) {
      console.warn('Failed to fetch TLE data from CelesTrak:', error);
      return null;
    }
  }

  /**
   * Update debris object with fresh TLE data
   */
  static async updateDebrisWithLiveTLE(debris: DebrisObject): Promise<DebrisObject> {
    if (!debris.tle) return debris;
    
    // Extract catalog number from existing TLE or use debris ID
    const catalogMatch = debris.tle.line1.match(/1 (\d{5})/);
    const catalogNumber = catalogMatch ? catalogMatch[1] : debris.id.replace(/\D/g, '').slice(-5);
    
    const liveTLE = await this.fetchFromCelesTrak(catalogNumber);
    
    if (liveTLE) {
      return {
        ...debris,
        tle: liveTLE,
        lastUpdate: new Date()
      };
    }
    
    return debris;
  }
}

// Export mock data for immediate use
export const MOCK_DEBRIS_DATA = DebrisDataGenerator.generateMockDebris(500);

export default {
  DebrisDataGenerator,
  SPACECRAFT_DATA,
  MOCK_DEBRIS_DATA,
  TLEDataFetcher,
};