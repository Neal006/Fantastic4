export type UserRole = 'admin' | 'operator';

export interface User {
  id: string;
  password: string;
  role: UserRole;
  name: string;
  email: string;
  assignedPlants?: string[];
  lastLogin?: string;
  status: 'active' | 'inactive';
}

export interface Plant {
  id: string;
  name: string;
  location: string;
  status: 'active' | 'maintenance' | 'decommissioned';
  blocks: Block[];
}

export interface Block {
  id: string;
  name: string;
  plantId: string;
  inverters: Inverter[];
}

export type InverterCategory = 'A' | 'B' | 'C' | 'D' | 'E' | 'offline';

export interface InverterReading {
  timestamp: string;
  dcVoltage: number;
  dcCurrent: number;
  acPower: number;
  moduleTemp: number;
  ambientTemp: number;
  irradiation: number;
}

export interface ShapValue {
  feature: string;
  value: number;
  label: string;
}

export interface FaultEvent {
  timestamp: string;
  category: InverterCategory;
  faultType: string;
}

export interface Inverter {
  id: string;
  blockId: string;
  plantId: string;
  capacity: number;
  category: InverterCategory;
  aiConfidence: number;
  faultType: string | null;
  lastUpdated: string;
  currentReadings: InverterReading;
  shapValues: ShapValue[];
  historicalReadings: InverterReading[];
  faultHistory: FaultEvent[];
}

export interface Alert {
  id: string;
  timestamp: string;
  inverterId: string;
  plantId: string;
  blockId: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  status: 'pending' | 'acknowledged';
  acknowledgedBy?: string;
  acknowledgedAt?: string;
}

export interface AuditLog {
  id: string;
  timestamp: string;
  userId: string;
  userName: string;
  role: UserRole;
  action: string;
  details: string;
  ipAddress: string;
}

export interface Settings {
  thresholds: {
    A: [number, number];
    B: [number, number];
    C: [number, number];
    D: [number, number];
    E: [number, number];
  };
  staleTimeout: number;
}
