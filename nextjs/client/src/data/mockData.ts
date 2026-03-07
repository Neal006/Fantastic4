import type { User, Plant, Block, Inverter, InverterCategory, InverterReading, ShapValue, FaultEvent, Alert, AuditLog, Settings } from './types';

// ── Helpers ──
const rng = (min: number, max: number) => Math.random() * (max - min) + min;
const rngInt = (min: number, max: number) => Math.floor(rng(min, max + 1));
const pick = <T>(arr: T[]): T => arr[rngInt(0, arr.length - 1)];
const uuid = () => crypto.randomUUID();
const ago = (mins: number) => new Date(Date.now() - mins * 60000).toISOString();

// ── Users ──
export const mockUsers: User[] = [
  { id: 'admin01', password: 'admin123', role: 'admin', name: 'Priya Sharma', email: 'priya@lumin.ai', assignedPlants: [], status: 'active', lastLogin: ago(12) },
  { id: 'operator01', password: 'oper123', role: 'operator', name: 'Rahul Kumar', email: 'rahul@lumin.ai', assignedPlants: ['plant-a', 'plant-b'], status: 'active', lastLogin: ago(5) },
  { id: 'operator02', password: 'oper456', role: 'operator', name: 'Sneha Patel', email: 'sneha@lumin.ai', assignedPlants: ['plant-a'], status: 'active', lastLogin: ago(60) },
];

// ── Readings Generator ──
function genReading(baseTime: Date, category: InverterCategory): InverterReading {
  const isOffline = category === 'offline';
  const dcV = isOffline ? 0 : rng(category === 'D' || category === 'E' ? 100 : 550, category === 'D' || category === 'E' ? 350 : 750);
  const dcI = isOffline ? 0 : rng(category === 'E' ? 0.1 : 5, category === 'D' ? 6 : 12);
  return {
    timestamp: baseTime.toISOString(),
    dcVoltage: +dcV.toFixed(1),
    dcCurrent: +dcI.toFixed(2),
    acPower: +(isOffline ? 0 : dcV * dcI * rng(0.85, 0.97) / 1000).toFixed(2),
    moduleTemp: +(isOffline ? 25 : rng(category === 'D' ? 65 : 30, category === 'E' ? 85 : 55)).toFixed(1),
    ambientTemp: +rng(28, 42).toFixed(1),
    irradiation: +(isOffline ? 0 : rng(category === 'E' ? 50 : 600, 1050)).toFixed(0),
  };
}

function genHistorical(category: InverterCategory): InverterReading[] {
  const readings: InverterReading[] = [];
  const now = Date.now();
  for (let i = 287; i >= 0; i--) {
    readings.push(genReading(new Date(now - i * 5 * 60000), category));
  }
  return readings;
}

function genShap(category: InverterCategory): ShapValue[] {
  if (category === 'A' || category === 'B') return [];
  const features = [
    { feature: 'dc_voltage', label: 'DC Voltage' },
    { feature: 'dc_current', label: 'DC Current' },
    { feature: 'ac_power', label: 'AC Power' },
    { feature: 'module_temp', label: 'Module Temperature' },
    { feature: 'ambient_temp', label: 'Ambient Temperature' },
    { feature: 'irradiation', label: 'Irradiation' },
  ];
  return features.map(f => ({
    ...f,
    value: +(category === 'E' ? rng(-0.3, 0.8) : rng(-0.2, 0.5)).toFixed(3),
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

const faultTypes: Record<string, string[]> = {
  C: ['Low Efficiency', 'Minor Voltage Deviation', 'Partial Shading'],
  D: ['String Failure', 'MPPT Drift', 'Overcurrent Warning'],
  E: ['Ground Fault', 'Arc Fault', 'Inverter Shutdown', 'DC Overcurrent'],
};

function assignCategory(): InverterCategory {
  const r = Math.random();
  if (r < 0.03) return 'offline';
  if (r < 0.08) return 'E';
  if (r < 0.13) return 'D';
  if (r < 0.28) return 'C';
  if (r < 0.55) return 'B';
  return 'A';
}

function genFaultHistory(category: InverterCategory): FaultEvent[] {
  if (category === 'A' || category === 'B') return [];
  const events: FaultEvent[] = [];
  const count = rngInt(1, 5);
  for (let i = 0; i < count; i++) {
    const cat = pick(['C', 'D', 'E'] as InverterCategory[]);
    events.push({
      timestamp: ago(rngInt(60, 10000)),
      category: cat,
      faultType: pick(faultTypes[cat] || ['Unknown']),
    });
  }
  return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

// ── Build Inverters ──
let invCounter = 0;
function genInverter(plantId: string, blockId: string): Inverter {
  invCounter++;
  const category = assignCategory();
  const ft = category === 'D' || category === 'E' ? pick(faultTypes[category]) : null;
  const stale = category === 'offline' ? rngInt(30, 120) : rngInt(1, 8);
  return {
    id: `INV-${String(invCounter).padStart(3, '0')}`,
    blockId,
    plantId,
    capacity: pick([50, 60, 75, 100]),
    category,
    aiConfidence: +(category === 'A' ? rng(0.92, 0.99) : category === 'B' ? rng(0.85, 0.94) : category === 'C' ? rng(0.6, 0.84) : rng(0.3, 0.6)).toFixed(2) as unknown as number,
    faultType: ft,
    lastUpdated: ago(stale),
    currentReadings: genReading(new Date(), category),
    shapValues: genShap(category),
    historicalReadings: genHistorical(category),
    faultHistory: genFaultHistory(category),
  };
}

// ── Build Plants ──
function buildBlock(plantId: string, name: string, invCount: number): Block {
  const blockId = `${plantId}-${name.toLowerCase().replace(/\s/g, '-')}`;
  const inverters: Inverter[] = [];
  for (let i = 0; i < invCount; i++) inverters.push(genInverter(plantId, blockId));
  return { id: blockId, name, plantId, inverters };
}

function buildPlant(id: string, name: string, location: string, blockDefs: { name: string; count: number }[]): Plant {
  return {
    id,
    name,
    location,
    status: 'active',
    blocks: blockDefs.map(b => buildBlock(id, b.name, b.count)),
  };
}

export let mockPlants: Plant[] = [
  buildPlant('plant-a', 'Rajasthan Solar Park', 'Jodhpur, Rajasthan', [
    { name: 'Block 1', count: 10 },
    { name: 'Block 2', count: 12 },
    { name: 'Block 3', count: 8 },
  ]),
  buildPlant('plant-b', 'Gujarat Sun Farm', 'Kutch, Gujarat', [
    { name: 'Block 1', count: 8 },
    { name: 'Block 2', count: 6 },
  ]),
  buildPlant('plant-c', 'Tamil Nadu Solar Grid', 'Ramanathapuram, TN', [
    { name: 'Block 1', count: 7 },
    { name: 'Block 2', count: 5 },
    { name: 'Block 3', count: 8 },
    { name: 'Block 4', count: 6 },
  ]),
];

// ── Alerts ──
function genAlerts(): Alert[] {
  const alerts: Alert[] = [];
  mockPlants.forEach(plant => {
    plant.blocks.forEach(block => {
      block.inverters.forEach(inv => {
        if (inv.category === 'D' || inv.category === 'E') {
          alerts.push({
            id: uuid(),
            timestamp: inv.lastUpdated,
            inverterId: inv.id,
            plantId: plant.id,
            blockId: block.id,
            severity: inv.category === 'E' ? 'critical' : 'warning',
            message: `${inv.faultType || 'Fault detected'} on ${inv.id}`,
            status: Math.random() > 0.5 ? 'acknowledged' : 'pending',
            acknowledgedBy: Math.random() > 0.5 ? 'Rahul Kumar' : undefined,
            acknowledgedAt: Math.random() > 0.5 ? ago(rngInt(1, 30)) : undefined,
          });
        }
        if (inv.category === 'C' && Math.random() > 0.6) {
          alerts.push({
            id: uuid(),
            timestamp: inv.lastUpdated,
            inverterId: inv.id,
            plantId: plant.id,
            blockId: block.id,
            severity: 'info',
            message: `Performance warning on ${inv.id}`,
            status: 'pending',
          });
        }
      });
    });
  });
  return alerts.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

export let mockAlerts: Alert[] = genAlerts();

// ── Audit Logs ──
export let mockAuditLogs: AuditLog[] = [
  { id: uuid(), timestamp: ago(5), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'create_plant', details: "Created plant 'Rajasthan Solar Park'", ipAddress: '192.168.1.10' },
  { id: uuid(), timestamp: ago(15), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'add_operator', details: "Added operator 'Rahul Kumar'", ipAddress: '192.168.1.10' },
  { id: uuid(), timestamp: ago(30), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'add_operator', details: "Added operator 'Sneha Patel'", ipAddress: '192.168.1.10' },
  { id: uuid(), timestamp: ago(45), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'update_settings', details: 'Updated category thresholds', ipAddress: '192.168.1.10' },
  { id: uuid(), timestamp: ago(120), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'create_plant', details: "Created plant 'Gujarat Sun Farm'", ipAddress: '192.168.1.12' },
  { id: uuid(), timestamp: ago(180), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'create_plant', details: "Created plant 'Tamil Nadu Solar Grid'", ipAddress: '192.168.1.12' },
  { id: uuid(), timestamp: ago(240), userId: 'operator01', userName: 'Rahul Kumar', role: 'operator', action: 'acknowledge_alert', details: 'Acknowledged alert for INV-005', ipAddress: '10.0.0.22' },
  { id: uuid(), timestamp: ago(300), userId: 'admin01', userName: 'Priya Sharma', role: 'admin', action: 'assign_plant', details: "Assigned 'Rajasthan Solar Park' to Rahul Kumar", ipAddress: '192.168.1.10' },
];

// ── Settings ──
export let mockSettings: Settings = {
  thresholds: {
    A: [90, 100],
    B: [75, 89],
    C: [50, 74],
    D: [25, 49],
    E: [0, 24],
  },
  staleTimeout: 10,
};

// ── Helper Functions ──
export function getAllInverters(): Inverter[] {
  return mockPlants.flatMap(p => p.blocks.flatMap(b => b.inverters));
}

export function getPlantName(plantId: string): string {
  return mockPlants.find(p => p.id === plantId)?.name || plantId;
}

export function getBlockName(blockId: string): string {
  for (const p of mockPlants) {
    const b = p.blocks.find(bl => bl.id === blockId);
    if (b) return b.name;
  }
  return blockId;
}

export function getInvertersByPlantAndBlock(plantId: string, blockId: string): Inverter[] {
  const plant = mockPlants.find(p => p.id === plantId);
  if (!plant) return [];
  const block = plant.blocks.find(b => b.id === blockId);
  return block?.inverters || [];
}

export function getPlantsForOperator(plantIds: string[]): Plant[] {
  return mockPlants.filter(p => plantIds.includes(p.id));
}

export function getAlertsForPlants(plantIds: string[]): Alert[] {
  return mockAlerts.filter(a => plantIds.includes(a.plantId));
}

export function acknowledgeAlert(alertId: string, userName: string) {
  mockAlerts = mockAlerts.map(a =>
    a.id === alertId ? { ...a, status: 'acknowledged' as const, acknowledgedBy: userName, acknowledgedAt: new Date().toISOString() } : a
  );
}

export function addAuditLog(entry: Omit<AuditLog, 'id'>) {
  mockAuditLogs = [{ id: uuid(), ...entry }, ...mockAuditLogs];
}
