-- ============================================================
-- SolarWatch Seed Data
-- Run after schema.sql
-- ============================================================

USE hackamined;

-- Clear existing data (order matters due to FK constraints)
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

-- ── Admins ──
-- Password: admin123
INSERT INTO admins (id, name, email, password_hash) VALUES
('admin-001', 'Priya Sharma', 'priya@solarwatch.in', '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY.5c0EiDPCKS6a');

-- ── Operators ──
-- operator01 / oper123
INSERT INTO operators (id, name, email, password_hash, is_active, created_by) VALUES
('op-001', 'Rahul Kumar', 'rahul@solarwatch.in', '$2a$12$nGHWfEWfimT9vCn1DliL4.2Bwi51qQr0fvfVtG3u1kgJHWGX..LZi', TRUE, 'admin-001'),
-- operator02 / oper456
('op-002', 'Sneha Patel', 'sneha@solarwatch.in', '$2a$12$vMYoYhDlCL3nJQQpZ5YGlua7yCe0mCIB7E2NKCq3FfZr1gkpBUhsi', TRUE, 'admin-001');

-- ── Plants ──
INSERT INTO plants (id, name, location, status, total_capacity_kw, created_by) VALUES
('plant-a', 'Rajasthan Solar Park', 'Jodhpur, Rajasthan', 'active', 5000, 'admin-001'),
('plant-b', 'Gujarat Sun Farm', 'Kutch, Gujarat', 'active', 3500, 'admin-001'),
('plant-c', 'Tamil Nadu Solar Grid', 'Ramanathapuram, TN', 'active', 4200, 'admin-001');

-- ── Operator Plant Access ──
INSERT INTO operator_plant_access (operator_id, plant_id) VALUES
('op-001', 'plant-a'),
('op-001', 'plant-b'),
('op-002', 'plant-a');

-- ── Blocks ──
INSERT INTO blocks (id, plant_id, name) VALUES
('plant-a-block-1', 'plant-a', 'Block 1'),
('plant-a-block-2', 'plant-a', 'Block 2'),
('plant-a-block-3', 'plant-a', 'Block 3'),
('plant-b-block-1', 'plant-b', 'Block 1'),
('plant-b-block-2', 'plant-b', 'Block 2'),
('plant-c-block-1', 'plant-c', 'Block 1'),
('plant-c-block-2', 'plant-c', 'Block 2'),
('plant-c-block-3', 'plant-c', 'Block 3'),
('plant-c-block-4', 'plant-c', 'Block 4');

-- ── Inverters ──
-- Plant A - Block 1 (10 inverters)
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
('inv-001', 'plant-a-block-1', 'INV-001', 'SN-PA-B1-001', 100, 'A', TRUE, NOW()),
('inv-002', 'plant-a-block-1', 'INV-002', 'SN-PA-B1-002', 100, 'B', TRUE, NOW()),
('inv-003', 'plant-a-block-1', 'INV-003', 'SN-PA-B1-003', 75,  'C', TRUE, NOW()),
('inv-004', 'plant-a-block-1', 'INV-004', 'SN-PA-B1-004', 100, 'A', TRUE, NOW()),
('inv-005', 'plant-a-block-1', 'INV-005', 'SN-PA-B1-005', 100, 'D', TRUE, NOW()),
('inv-006', 'plant-a-block-1', 'INV-006', 'SN-PA-B1-006', 75,  'A', TRUE, NOW()),
('inv-007', 'plant-a-block-1', 'INV-007', 'SN-PA-B1-007', 100, 'E', TRUE, NOW()),
('inv-008', 'plant-a-block-1', 'INV-008', 'SN-PA-B1-008', 50,  'A', TRUE, NOW()),
('inv-009', 'plant-a-block-1', 'INV-009', 'SN-PA-B1-009', 100, 'B', TRUE, NOW()),
('inv-010', 'plant-a-block-1', 'INV-010', 'SN-PA-B1-010', 75,  'offline', FALSE, DATE_SUB(NOW(), INTERVAL 20 MINUTE));

-- Plant A - Block 2 (12 inverters)
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
('inv-011', 'plant-a-block-2', 'INV-011', 'SN-PA-B2-001', 100, 'A', TRUE, NOW()),
('inv-012', 'plant-a-block-2', 'INV-012', 'SN-PA-B2-002', 100, 'A', TRUE, NOW()),
('inv-013', 'plant-a-block-2', 'INV-013', 'SN-PA-B2-003', 100, 'B', TRUE, NOW()),
('inv-014', 'plant-a-block-2', 'INV-014', 'SN-PA-B2-004', 75,  'C', TRUE, NOW()),
('inv-015', 'plant-a-block-2', 'INV-015', 'SN-PA-B2-005', 100, 'A', TRUE, NOW()),
('inv-016', 'plant-a-block-2', 'INV-016', 'SN-PA-B2-006', 100, 'A', TRUE, NOW()),
('inv-017', 'plant-a-block-2', 'INV-017', 'SN-PA-B2-007', 50,  'D', TRUE, NOW()),
('inv-018', 'plant-a-block-2', 'INV-018', 'SN-PA-B2-008', 100, 'A', TRUE, NOW()),
('inv-019', 'plant-a-block-2', 'INV-019', 'SN-PA-B2-009', 75,  'B', TRUE, NOW()),
('inv-020', 'plant-a-block-2', 'INV-020', 'SN-PA-B2-010', 100, 'A', TRUE, NOW()),
('inv-021', 'plant-a-block-2', 'INV-021', 'SN-PA-B2-011', 100, 'C', TRUE, NOW()),
('inv-022', 'plant-a-block-2', 'INV-022', 'SN-PA-B2-012', 75,  'A', TRUE, NOW());

-- Plant A - Block 3 (8 inverters)
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
('inv-023', 'plant-a-block-3', 'INV-023', 'SN-PA-B3-001', 100, 'A', TRUE, NOW()),
('inv-024', 'plant-a-block-3', 'INV-024', 'SN-PA-B3-002', 100, 'B', TRUE, NOW()),
('inv-025', 'plant-a-block-3', 'INV-025', 'SN-PA-B3-003', 75,  'E', TRUE, NOW()),
('inv-026', 'plant-a-block-3', 'INV-026', 'SN-PA-B3-004', 100, 'A', TRUE, NOW()),
('inv-027', 'plant-a-block-3', 'INV-027', 'SN-PA-B3-005', 100, 'A', TRUE, NOW()),
('inv-028', 'plant-a-block-3', 'INV-028', 'SN-PA-B3-006', 75,  'C', TRUE, NOW()),
('inv-029', 'plant-a-block-3', 'INV-029', 'SN-PA-B3-007', 100, 'D', TRUE, NOW()),
('inv-030', 'plant-a-block-3', 'INV-030', 'SN-PA-B3-008', 50,  'A', TRUE, NOW());

-- Plant B - Block 1 (8 inverters)
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
('inv-031', 'plant-b-block-1', 'INV-031', 'SN-PB-B1-001', 60, 'A', TRUE, NOW()),
('inv-032', 'plant-b-block-1', 'INV-032', 'SN-PB-B1-002', 60, 'B', TRUE, NOW()),
('inv-033', 'plant-b-block-1', 'INV-033', 'SN-PB-B1-003', 60, 'A', TRUE, NOW()),
('inv-034', 'plant-b-block-1', 'INV-034', 'SN-PB-B1-004', 60, 'C', TRUE, NOW()),
('inv-035', 'plant-b-block-1', 'INV-035', 'SN-PB-B1-005', 60, 'A', TRUE, NOW()),
('inv-036', 'plant-b-block-1', 'INV-036', 'SN-PB-B1-006', 60, 'A', TRUE, NOW()),
('inv-037', 'plant-b-block-1', 'INV-037', 'SN-PB-B1-007', 60, 'D', TRUE, NOW()),
('inv-038', 'plant-b-block-1', 'INV-038', 'SN-PB-B1-008', 60, 'offline', FALSE, DATE_SUB(NOW(), INTERVAL 15 MINUTE));

-- Plant B - Block 2 (6 inverters)
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
('inv-039', 'plant-b-block-2', 'INV-039', 'SN-PB-B2-001', 60, 'A', TRUE, NOW()),
('inv-040', 'plant-b-block-2', 'INV-040', 'SN-PB-B2-002', 60, 'B', TRUE, NOW()),
('inv-041', 'plant-b-block-2', 'INV-041', 'SN-PB-B2-003', 60, 'A', TRUE, NOW()),
('inv-042', 'plant-b-block-2', 'INV-042', 'SN-PB-B2-004', 60, 'A', TRUE, NOW()),
('inv-043', 'plant-b-block-2', 'INV-043', 'SN-PB-B2-005', 60, 'E', TRUE, NOW()),
('inv-044', 'plant-b-block-2', 'INV-044', 'SN-PB-B2-006', 60, 'A', TRUE, NOW());

-- Plant C inverters (blocks 1-4)
INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw, current_category, is_online, last_data_at) VALUES
('inv-045', 'plant-c-block-1', 'INV-045', 'SN-PC-B1-001', 75, 'A', TRUE, NOW()),
('inv-046', 'plant-c-block-1', 'INV-046', 'SN-PC-B1-002', 75, 'B', TRUE, NOW()),
('inv-047', 'plant-c-block-1', 'INV-047', 'SN-PC-B1-003', 75, 'A', TRUE, NOW()),
('inv-048', 'plant-c-block-1', 'INV-048', 'SN-PC-B1-004', 75, 'C', TRUE, NOW()),
('inv-049', 'plant-c-block-1', 'INV-049', 'SN-PC-B1-005', 75, 'A', TRUE, NOW()),
('inv-050', 'plant-c-block-1', 'INV-050', 'SN-PC-B1-006', 75, 'D', TRUE, NOW()),
('inv-051', 'plant-c-block-1', 'INV-051', 'SN-PC-B1-007', 75, 'A', TRUE, NOW()),
('inv-052', 'plant-c-block-2', 'INV-052', 'SN-PC-B2-001', 75, 'A', TRUE, NOW()),
('inv-053', 'plant-c-block-2', 'INV-053', 'SN-PC-B2-002', 75, 'E', TRUE, NOW()),
('inv-054', 'plant-c-block-2', 'INV-054', 'SN-PC-B2-003', 75, 'A', TRUE, NOW()),
('inv-055', 'plant-c-block-2', 'INV-055', 'SN-PC-B2-004', 75, 'B', TRUE, NOW()),
('inv-056', 'plant-c-block-2', 'INV-056', 'SN-PC-B2-005', 75, 'A', TRUE, NOW()),
('inv-057', 'plant-c-block-3', 'INV-057', 'SN-PC-B3-001', 75, 'A', TRUE, NOW()),
('inv-058', 'plant-c-block-3', 'INV-058', 'SN-PC-B3-002', 75, 'C', TRUE, NOW()),
('inv-059', 'plant-c-block-3', 'INV-059', 'SN-PC-B3-003', 75, 'A', TRUE, NOW()),
('inv-060', 'plant-c-block-3', 'INV-060', 'SN-PC-B3-004', 75, 'A', TRUE, NOW()),
('inv-061', 'plant-c-block-3', 'INV-061', 'SN-PC-B3-005', 75, 'D', TRUE, NOW()),
('inv-062', 'plant-c-block-3', 'INV-062', 'SN-PC-B3-006', 75, 'A', TRUE, NOW()),
('inv-063', 'plant-c-block-3', 'INV-063', 'SN-PC-B3-007', 75, 'A', TRUE, NOW()),
('inv-064', 'plant-c-block-3', 'INV-064', 'SN-PC-B3-008', 75, 'B', TRUE, NOW()),
('inv-065', 'plant-c-block-4', 'INV-065', 'SN-PC-B4-001', 75, 'A', TRUE, NOW()),
('inv-066', 'plant-c-block-4', 'INV-066', 'SN-PC-B4-002', 75, 'A', TRUE, NOW()),
('inv-067', 'plant-c-block-4', 'INV-067', 'SN-PC-B4-003', 75, 'E', TRUE, NOW()),
('inv-068', 'plant-c-block-4', 'INV-068', 'SN-PC-B4-004', 75, 'A', TRUE, NOW()),
('inv-069', 'plant-c-block-4', 'INV-069', 'SN-PC-B4-005', 75, 'C', TRUE, NOW()),
('inv-070', 'plant-c-block-4', 'INV-070', 'SN-PC-B4-006', 75, 'A', TRUE, NOW());

-- ── Seed some alerts ──
INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to, acknowledged, acknowledged_by, acknowledged_at) VALUES
(UUID(), 'inv-007', 'critical', 'Arc Fault detected on INV-007', 'D', 'E', FALSE, NULL, NULL),
(UUID(), 'inv-005', 'warning', 'MPPT Drift detected on INV-005', 'C', 'D', FALSE, NULL, NULL),
(UUID(), 'inv-025', 'critical', 'Ground Fault detected on INV-025', 'D', 'E', TRUE, 'op-001', DATE_SUB(NOW(), INTERVAL 5 MINUTE)),
(UUID(), 'inv-017', 'warning', 'String Failure on INV-017', 'C', 'D', FALSE, NULL, NULL),
(UUID(), 'inv-037', 'warning', 'Overcurrent Warning on INV-037', 'C', 'D', TRUE, 'op-001', DATE_SUB(NOW(), INTERVAL 10 MINUTE)),
(UUID(), 'inv-043', 'critical', 'Inverter Shutdown on INV-043', 'D', 'E', FALSE, NULL, NULL),
(UUID(), 'inv-053', 'critical', 'DC Overcurrent on INV-053', 'D', 'E', FALSE, NULL, NULL),
(UUID(), 'inv-067', 'critical', 'Arc Fault on INV-067', 'D', 'E', FALSE, NULL, NULL),
(UUID(), 'inv-050', 'warning', 'MPPT Drift on INV-050', 'C', 'D', TRUE, 'op-002', DATE_SUB(NOW(), INTERVAL 20 MINUTE)),
(UUID(), 'inv-029', 'warning', 'String Failure on INV-029', 'C', 'D', FALSE, NULL, NULL);

-- ── Seed Audit Logs ──
INSERT INTO audit_logs (id, user_id, user_role, action, details, ip_address) VALUES
(UUID(), 'admin-001', 'admin', 'create_plant', '{"plant_name":"Rajasthan Solar Park"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'create_plant', '{"plant_name":"Gujarat Sun Farm"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'create_plant', '{"plant_name":"Tamil Nadu Solar Grid"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'add_operator', '{"operator_name":"Rahul Kumar","email":"rahul@solarwatch.in"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'add_operator', '{"operator_name":"Sneha Patel","email":"sneha@solarwatch.in"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'assign_plant', '{"operator":"Rahul Kumar","plant":"Rajasthan Solar Park"}', '192.168.1.10'),
(UUID(), 'admin-001', 'admin', 'assign_plant', '{"operator":"Rahul Kumar","plant":"Gujarat Sun Farm"}', '192.168.1.10'),
(UUID(), 'op-001', 'operator', 'acknowledge_alert', '{"inverter":"INV-025","fault_type":"Ground Fault"}', '10.0.0.22');

-- Confirm
SELECT 'Seed complete!' AS status;
SELECT COUNT(*) AS inverter_count FROM inverters;
SELECT COUNT(*) AS alert_count FROM alerts;
