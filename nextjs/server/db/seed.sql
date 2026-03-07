-- ============================================================
-- SolarWatch Seed Data  —  Synthetic Inverter Dataset
-- Matches genai/app/synthetic_data.py exactly (12 inverters, 3 plants)
-- Run after schema.sql
-- ============================================================

USE hackamined;

-- Clear existing data (FK-safe order)
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE audit_logs;
TRUNCATE TABLE alerts;
TRUNCATE TABLE inverter_readings;
TRUNCATE TABLE inverters;
TRUNCATE TABLE operator_plant_access;
TRUNCATE TABLE blocks;
TRUNCATE TABLE plants;
TRUNCATE TABLE operators;
TRUNCATE TABLE admins;
SET FOREIGN_KEY_CHECKS = 1;

-- ── Admins ───────────────────────────────────────────────────
-- Password: admin123
INSERT INTO admins (id, name, email, password_hash) VALUES
('admin-001', 'Priya Sharma', 'priya@solarwatch.in', '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY.5c0EiDPCKS6a');

-- ── Operators ────────────────────────────────────────────────
-- OP001 / operator123
INSERT INTO operators (id, name, email, password_hash, is_active, created_by) VALUES
('op-001', 'Rahul Kumar',  'rahul@solarwatch.in', '$2a$12$nGHWfEWfimT9vCn1DliL4.2Bwi51qQr0fvfVtG3u1kgJHWGX..LZi', TRUE, 'admin-001'),
-- OP002 / operator456
('op-002', 'Sneha Patel',  'sneha@solarwatch.in', '$2a$12$vMYoYhDlCL3nJQQpZ5YGlua7yCe0mCIB7E2NKCq3FfZr1gkpBUhsi', TRUE, 'admin-001');

-- ── Plants  (match synthetic_data.py PLANTS dict) ────────────
-- plant_1 → Plant 1 - Celestical
-- plant_2 → Plant 2
-- plant_3 → Plant 3
INSERT INTO plants (id, name, location, status, total_capacity_kw, created_by) VALUES
('plant-1', 'Plant 1 - Celestical', 'Rajasthan, India',  'active', 500, 'admin-001'),
('plant-2', 'Plant 2',              'Gujarat, India',    'active', 500, 'admin-001'),
('plant-3', 'Plant 3',              'Tamil Nadu, India', 'active', 500, 'admin-001');

-- ── Operator Plant Access ────────────────────────────────────
INSERT INTO operator_plant_access (operator_id, plant_id) VALUES
('op-001', 'plant-1'),
('op-001', 'plant-2'),
('op-002', 'plant-2'),
('op-002', 'plant-3');

-- ── Blocks  (2 per plant: Block A and Block B) ───────────────
INSERT INTO blocks (id, plant_id, name) VALUES
('p1-block-a', 'plant-1', 'Block A'),   -- ICR2-LT1 / INV-P1-L1-*
('p1-block-b', 'plant-1', 'Block B'),   -- ICR2-LT2 / INV-P1-L2-*
('p2-block-a', 'plant-2', 'Block A'),   -- Logger-AC12 / INV-P2-L1-*
('p2-block-b', 'plant-2', 'Block B'),   -- Logger-ACBB / INV-P2-L2-*
('p3-block-a', 'plant-3', 'Block A'),   -- Logger-1469 / INV-P3-L1-*
('p3-block-b', 'plant-3', 'Block B');   -- Logger-146E / INV-P3-L2-*

-- ── Inverters  (12 total — names match GenAI synthetic inverter IDs) ──
-- Category derived from risk score:
--   A = 0–12%  B = 13–24%  C = 25–64%  D = 65–79%  E = 80–100%
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
-- Plant 1 / Block A
('p1a-inv-0', 'p1-block-a', 'INV-P1-L1-0', 'ICR2-LT1-0', 100, 'A', TRUE, NOW()),  -- risk 12%
('p1a-inv-1', 'p1-block-a', 'INV-P1-L1-1', 'ICR2-LT1-1', 100, 'D', TRUE, NOW()),  -- risk 65%
-- Plant 1 / Block B
('p1b-inv-0', 'p1-block-b', 'INV-P1-L2-0', 'ICR2-LT2-0', 100, 'E', TRUE, NOW()),  -- risk 89%  CRITICAL
('p1b-inv-1', 'p1-block-b', 'INV-P1-L2-1', 'ICR2-LT2-1', 100, 'A', TRUE, NOW()),  -- risk  8%
-- Plant 2 / Block A
('p2a-inv-0', 'p2-block-a', 'INV-P2-L1-0', 'ACBB-L1-0',  100, 'D', TRUE, NOW()),  -- risk 72%
('p2a-inv-1', 'p2-block-a', 'INV-P2-L1-1', 'ACBB-L1-1',  100, 'B', TRUE, NOW()),  -- risk 15%
-- Plant 2 / Block B
('p2b-inv-0', 'p2-block-b', 'INV-P2-L2-0', 'ACBB-L2-0',  100, 'C', TRUE, NOW()),  -- risk 58%
('p2b-inv-1', 'p2-block-b', 'INV-P2-L2-1', 'ACBB-L2-1',  100, 'E', FALSE, DATE_SUB(NOW(), INTERVAL 5 MINUTE)),  -- risk 91%  CRITICAL / grid fault
-- Plant 3 / Block A
('p3a-inv-0', 'p3-block-a', 'INV-P3-L1-0', '1469-L1-0',  100, 'A', TRUE, NOW()),  -- risk  5%
('p3a-inv-1', 'p3-block-a', 'INV-P3-L1-1', '1469-L1-1',  100, 'A', TRUE, NOW()),  -- risk 11%
-- Plant 3 / Block B
('p3b-inv-0', 'p3-block-b', 'INV-P3-L2-0', '146E-L2-0',  100, 'C', TRUE, NOW()),  -- risk 45%
('p3b-inv-1', 'p3-block-b', 'INV-P3-L2-1', '146E-L2-1',  100, 'C', TRUE, NOW());  -- risk 52%

-- ── Initial inverter_readings  (one seed reading per inverter) ──
-- dc_voltage  = pv1_voltage (V)
-- dc_current  = pv1_current (A)
-- ac_power    = power   (kW)    — raw is Watts so /1000
-- module_temp = inverters[x].temp (°C)
-- ambient_temp= sensors[0].ambient_temp (°C)
-- irradiation = derived from power / capacity (W/m²) — use sensible placeholder
-- shap_values = JSON array of {label, value} for the top SHAP features

-- INV-P1-L1-0 : normal operation (risk 12%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p1a-inv-0', NOW(), 'A', 0.9700, FALSE, NULL,
  38.20, 9.80, 8.750, 42.3, 34.2, 920.0,
  '[{"label":"power","value":-0.04},{"label":"temp","value":-0.03},{"label":"ambient_temp","value":-0.02},{"label":"pv1_current","value":-0.01},{"label":"pf","value":-0.01}]');

-- INV-P1-L1-1 : string degradation (risk 65%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p1a-inv-1', NOW(), 'D', 0.6500, TRUE, 'String Degradation',
  37.80, 9.50, 6.200, 44.1, 35.8, 820.0,
  '[{"label":"pv3_current","value":0.18},{"label":"string3","value":0.14},{"label":"power","value":0.12},{"label":"pv3_voltage","value":0.08},{"label":"ambient_temp","value":0.04}]');

-- INV-P1-L2-0 : overheating — SHUTDOWN RISK (risk 89%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p1b-inv-0', NOW(), 'E', 0.8900, TRUE, 'Overheating — Thermal Shutdown Risk',
  36.50, 8.10, 4.100, 78.6, 47.3, 750.0,
  '[{"label":"temp","value":0.35},{"label":"ambient_temp","value":0.15},{"label":"alarm_code","value":0.12},{"label":"power","value":0.10},{"label":"limit_percent","value":0.08}]');

-- INV-P1-L2-1 : normal operation (risk 8%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p1b-inv-1', NOW(), 'A', 0.9800, FALSE, NULL,
  38.50, 9.90, 9.100, 40.5, 33.8, 950.0,
  NULL);

-- INV-P2-L1-0 : alarm triggered (risk 72%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p2a-inv-0', NOW(), 'D', 0.7200, TRUE, 'Alarm Code 3021 — Operational Fault',
  35.10, 6.80, 5.500, 51.2, 36.5, 700.0,
  '[{"label":"alarm_code","value":0.28},{"label":"op_state","value":0.18},{"label":"power","value":0.12},{"label":"pv1_current","value":0.06},{"label":"meter_active_power","value":0.05}]');

-- INV-P2-L1-1 : normal operation (risk 15%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p2a-inv-1', NOW(), 'B', 0.9200, FALSE, NULL,
  38.00, 9.60, 8.900, 41.8, 35.1, 910.0,
  NULL);

-- INV-P2-L2-0 : low power output (risk 58%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p2b-inv-0', NOW(), 'C', 0.5800, TRUE, 'Low Power Output — String Issue',
  37.50, 9.30, 5.100, 43.7, 36.0, 780.0,
  '[{"label":"power","value":0.22},{"label":"pv5_current","value":0.09},{"label":"pv6_current","value":0.08},{"label":"kwh_today","value":0.07},{"label":"pf","value":0.05}]');

-- INV-P2-L2-1 : grid fault — SHUTDOWN RISK (risk 91%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p2b-inv-1', DATE_SUB(NOW(), INTERVAL 5 MINUTE), 'E', 0.9100, TRUE, 'Grid Fault — Frequency Deviation 51.8 Hz',
  38.10, 0.00, 0.000, 38.2, 34.5, 200.0,
  '[{"label":"freq","value":0.30},{"label":"v_r","value":0.20},{"label":"alarm_code","value":0.15},{"label":"op_state","value":0.12},{"label":"pf","value":0.08}]');

-- INV-P3-L1-0 : normal operation (risk 5%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p3a-inv-0', NOW(), 'A', 0.9900, FALSE, NULL,
  38.60, 9.90, 9.200, 39.8, 32.1, 960.0,
  NULL);

-- INV-P3-L1-1 : normal operation (risk 11%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p3a-inv-1', NOW(), 'A', 0.9700, FALSE, NULL,
  38.10, 9.70, 8.800, 41.0, 32.5, 930.0,
  NULL);

-- INV-P3-L2-0 : partial shading (risk 45%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p3b-inv-0', NOW(), 'C', 0.4500, TRUE, 'Partial Shading — Strings 7 & 8',
  37.80, 9.40, 7.100, 42.0, 33.5, 800.0,
  '[{"label":"string7","value":0.12},{"label":"string8","value":0.10},{"label":"pv7_current","value":0.08},{"label":"pv8_current","value":0.06},{"label":"power","value":0.05}]');

-- INV-P3-L2-1 : communication issue (risk 52%)
INSERT INTO inverter_readings (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
  dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values) VALUES
(UUID(), 'p3b-inv-1', NOW(), 'C', 0.5200, TRUE, 'Communication Issue — Alarm Code 2010',
  37.00, 8.80, 6.800, 43.5, 34.0, 830.0,
  '[{"label":"op_state","value":0.18},{"label":"alarm_code","value":0.12},{"label":"power","value":0.10},{"label":"string1","value":0.05},{"label":"string2","value":0.04}]');

-- ── Alerts  (for critical & warning inverters) ────────────────
INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to, acknowledged) VALUES
(UUID(), 'p1b-inv-0', 'critical', 'Thermal Shutdown Risk on INV-P1-L2-0 — temp 78.6°C (alarm 4003)', 'D', 'E', FALSE),
(UUID(), 'p2b-inv-1', 'critical', 'Grid Fault on INV-P2-L2-1 — freq 51.8 Hz, voltage deviation detected', 'D', 'E', FALSE),
(UUID(), 'p1a-inv-1', 'warning',  'String Degradation on INV-P1-L1-1 — pv3_current 3.1A (expected ~9.5A)', 'C', 'D', FALSE),
(UUID(), 'p2a-inv-0', 'warning',  'Alarm Code 3021 on INV-P2-L1-0 — power limited to 60%', 'C', 'D', FALSE),
(UUID(), 'p2b-inv-0', 'info',     'Low Power Output on INV-P2-L2-0 — pv5/pv6 strings underperforming', 'B', 'C', FALSE),
(UUID(), 'p3b-inv-0', 'info',     'Partial Shading on INV-P3-L2-0 — strings 7 & 8 reduced output', 'B', 'C', FALSE),
(UUID(), 'p3b-inv-1', 'info',     'Communication Issue on INV-P3-L2-1 — alarm code 2010, op_state 4', 'B', 'C', FALSE);

-- ── Audit Logs ───────────────────────────────────────────────
INSERT INTO audit_logs (id, user_id, user_role, action, details, ip_address) VALUES
(UUID(), 'admin-001', 'admin', 'create_plant', '{"plant_name":"Plant 1 - Celestical"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'create_plant', '{"plant_name":"Plant 2"}',               '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'create_plant', '{"plant_name":"Plant 3"}',               '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'add_operator', '{"operator_name":"Rahul Kumar","email":"rahul@solarwatch.in"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'add_operator', '{"operator_name":"Sneha Patel","email":"sneha@solarwatch.in"}',  '192.168.1.10');

-- ── Confirm ──────────────────────────────────────────────────
SELECT 'Seed complete!' AS status;
SELECT COUNT(*) AS inverter_count   FROM inverters;
SELECT COUNT(*) AS block_count      FROM blocks;
SELECT COUNT(*) AS plant_count      FROM plants;
SELECT COUNT(*) AS reading_count    FROM inverter_readings;
SELECT COUNT(*) AS alert_count      FROM alerts;
