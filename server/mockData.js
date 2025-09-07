// Mock data generation utilities for the backend
const MOCK_TLE_DATABASE = {
  // Sample TLE data for various debris types
  'COSMOS-2251': {
    line1: '1 22675U 93036SX  24001.50000000  .00000000  00000-0  00000-0 0    07',
    line2: '2 22675  74.0458 280.7864 0012345 123.4567 236.8901 14.12345678123456'
  },
  'IRIDIUM-33': {
    line1: '1 24946U 97051C   24001.50000000  .00000000  00000-0  00000-0 0    08',
    line2: '2 24946  86.4008 123.7864 0002345 234.5678 125.4321 14.34567891234567'
  },
  'FENGYUN-1C': {
    line1: '1 25730U 99025A   24001.50000000  .00000000  00000-0  00000-0 0    09',
    line2: '2 25730  98.7654 187.6543 0001234 156.7890 203.4567 14.89012345678901'
  }
};

function generateMockDebris(count = 500) {
  const debris = [];
  const sourceObjects = Object.keys(MOCK_TLE_DATABASE);
  
  for (let i = 0; i < count; i++) {
    const id = `DEBRIS-${String(1000 + i).padStart(4, '0')}`;
    const sourceObject = sourceObjects[Math.floor(Math.random() * sourceObjects.length)];
    
    // Generate realistic orbital parameters
    const altitude = generateRealisticAltitude();
    const inclination = generateInclination();
    const eccentricity = Math.random() * 0.1;
    
    // Position calculation
    const orbitRadius = 6371 + altitude;
    const theta = Math.random() * 2 * Math.PI;
    const phi = Math.acos(2 * Math.random() - 1);
    
    const position = {
      x: orbitRadius * Math.sin(phi) * Math.cos(theta),
      y: orbitRadius * Math.sin(phi) * Math.sin(theta),
      z: orbitRadius * Math.cos(phi)
    };

    // Velocity calculation
    const orbitalVelocity = Math.sqrt(398600.4418 / orbitRadius);
    const velocity = {
      x: -orbitalVelocity * Math.sin(theta) + (Math.random() - 0.5) * 0.5,
      y: orbitalVelocity * Math.cos(theta) + (Math.random() - 0.5) * 0.5,
      z: (Math.random() - 0.5) * 0.2
    };

    const size = generateDebrisSize();
    const mass = calculateMass(size);
    const riskLevel = getWeightedRiskLevel();

    debris.push({
      id,
      name: generateDebrisName(sourceObject, i),
      position,
      velocity,
      size,
      mass,
      riskLevel,
      riskScore: calculateRiskScore(riskLevel, size, altitude),
      altitude,
      inclination,
      eccentricity,
      lastUpdate: new Date(),
      tle: generateTLEData(id, altitude, inclination, eccentricity),
      nextCloseApproach: Math.random() > 0.8 ? generateCloseApproach() : undefined
    });
  }

  return debris;
}

function generateRealisticAltitude() {
  const random = Math.random();
  
  if (random < 0.4) {
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

function generateInclination() {
  const random = Math.random();
  
  if (random < 0.4) {
    return 98 + Math.random() * 2; // Sun-synchronous
  } else if (random < 0.6) {
    return 85 + Math.random() * 10; // Polar
  } else if (random < 0.8) {
    return 30 + Math.random() * 30; // Medium inclination
  } else {
    return Math.random() * 30; // Low inclination
  }
}

function generateDebrisSize() {
  const random = Math.random();
  
  if (random < 0.7) {
    return 0.01 + Math.random() * 0.09; // 1cm - 10cm
  } else if (random < 0.9) {
    return 0.1 + Math.random() * 0.9; // 10cm - 1m
  } else {
    return 1 + Math.random() * 9; // 1m - 10m
  }
}

function calculateMass(size) {
  const volume = (4/3) * Math.PI * Math.pow(size/2, 3);
  return volume * 2800; // Aluminum density
}

function getWeightedRiskLevel() {
  const random = Math.random();
  
  if (random < 0.6) return 'low';
  if (random < 0.85) return 'medium';
  if (random < 0.97) return 'high';
  return 'critical';
}

function calculateRiskScore(riskLevel, size, altitude) {
  let baseScore = 0;
  
  switch (riskLevel) {
    case 'low': baseScore = 10 + Math.random() * 20; break;
    case 'medium': baseScore = 30 + Math.random() * 20; break;
    case 'high': baseScore = 50 + Math.random() * 30; break;
    case 'critical': baseScore = 80 + Math.random() * 20; break;
  }

  const sizeFactor = Math.min(size * 10, 20);
  const altitudeFactor = altitude < 800 ? 10 : altitude < 2000 ? 5 : 0;
  
  return Math.min(Math.round(baseScore + sizeFactor + altitudeFactor), 100);
}

function generateDebrisName(source, index) {
  const suffixes = ['DEB', 'FRAG', 'PART', 'PIECE', 'SEC'];
  const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
  return `${source}-${String(index).padStart(3, '0')}-${suffix}`;
}

function generateTLEData(id, altitude, inclination, eccentricity) {
  const catalogNumber = String(Math.floor(Math.random() * 99999)).padStart(5, '0');
  const epochYear = '24';
  const epochDay = (Math.random() * 365 + 1).toFixed(8).padStart(12, '0');
  
  const semiMajorAxis = 6371 + altitude;
  const meanMotion = Math.sqrt(398600.4418 / (semiMajorAxis ** 3)) * 86400 / (2 * Math.PI);
  
  const line1 = `1 ${catalogNumber}U 24001A   ${epochYear}${epochDay}  .00000000  00000-0  00000-0 0    0${Math.floor(Math.random() * 10)}`;
  const line2 = `2 ${catalogNumber} ${inclination.toFixed(4).padStart(8)} ${(Math.random() * 360).toFixed(4).padStart(8)} ${eccentricity.toFixed(7)} ${(Math.random() * 360).toFixed(4).padStart(8)} ${(Math.random() * 360).toFixed(4).padStart(8)} ${meanMotion.toFixed(8)}${String(Math.floor(Math.random() * 99999)).padStart(5, '0')}`;
  
  return { line1, line2 };
}

function generateCloseApproach() {
  const targets = ['ISS-ZARYA', 'TIANGONG', 'HUBBLE'];
  return {
    targetId: targets[Math.floor(Math.random() * targets.length)],
    distance: Math.random() * 50 + 5,
    time: new Date(Date.now() + Math.random() * 7 * 24 * 60 * 60 * 1000)
  };
}

module.exports = {
  generateMockDebris,
  MOCK_TLE_DATABASE
};