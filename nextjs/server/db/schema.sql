-- ============================================================
-- SolarWatch Database Schema
-- Database: hackamined
-- Run this file to create all tables
-- ============================================================

USE hackamined;

-- Admins
CREATE TABLE IF NOT EXISTS admins (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  name VARCHAR(100) NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Operators
CREATE TABLE IF NOT EXISTS operators (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  name VARCHAR(100) NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  must_reset_password BOOLEAN DEFAULT FALSE,
  created_by VARCHAR(36),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_login TIMESTAMP NULL,
  FOREIGN KEY (created_by) REFERENCES admins(id) ON DELETE SET NULL
);

-- Plants
CREATE TABLE IF NOT EXISTS plants (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  name VARCHAR(150) NOT NULL,
  location VARCHAR(200) NOT NULL,
  status ENUM('active', 'maintenance', 'decommissioned') DEFAULT 'active',
  total_capacity_kw DECIMAL(10,2) DEFAULT 0,
  created_by VARCHAR(36),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES admins(id) ON DELETE SET NULL
);

-- Operator Plant Access
CREATE TABLE IF NOT EXISTS operator_plant_access (
  operator_id VARCHAR(36) NOT NULL,
  plant_id VARCHAR(36) NOT NULL,
  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (operator_id, plant_id),
  FOREIGN KEY (operator_id) REFERENCES operators(id) ON DELETE CASCADE,
  FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
);

-- Blocks
CREATE TABLE IF NOT EXISTS blocks (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  plant_id VARCHAR(36) NOT NULL,
  name VARCHAR(100) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
);

-- Inverters
CREATE TABLE IF NOT EXISTS inverters (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  block_id VARCHAR(36) NOT NULL,
  name VARCHAR(100) NOT NULL,
  serial_number VARCHAR(100),
  capacity_kw DECIMAL(8,2) DEFAULT 50,
  current_category ENUM('A', 'B', 'C', 'D', 'E', 'offline') DEFAULT 'A',
  is_online BOOLEAN DEFAULT TRUE,
  last_data_at TIMESTAMP NULL,
  installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
);

-- Inverter Readings
CREATE TABLE IF NOT EXISTS inverter_readings (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  inverter_id VARCHAR(36) NOT NULL,
  timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  category ENUM('A', 'B', 'C', 'D', 'E', 'offline') NOT NULL,
  confidence DECIMAL(5,4) DEFAULT 0.90,
  is_faulty BOOLEAN DEFAULT FALSE,
  fault_type VARCHAR(100) NULL,
  dc_voltage DECIMAL(8,2) DEFAULT 0,
  dc_current DECIMAL(8,4) DEFAULT 0,
  ac_power DECIMAL(10,3) DEFAULT 0,
  module_temp DECIMAL(6,2) DEFAULT 0,
  ambient_temp DECIMAL(6,2) DEFAULT 0,
  irradiation DECIMAL(8,2) DEFAULT 0,
  shap_values JSON NULL,
  FOREIGN KEY (inverter_id) REFERENCES inverters(id) ON DELETE CASCADE,
  INDEX idx_inverter_timestamp (inverter_id, timestamp)
);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  inverter_id VARCHAR(36) NOT NULL,
  type ENUM('warning', 'critical', 'info') NOT NULL,
  message TEXT NOT NULL,
  category_from ENUM('A', 'B', 'C', 'D', 'E', 'offline') NULL,
  category_to ENUM('A', 'B', 'C', 'D', 'E', 'offline') NULL,
  acknowledged BOOLEAN DEFAULT FALSE,
  acknowledged_by VARCHAR(36) NULL,
  acknowledged_at TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (inverter_id) REFERENCES inverters(id) ON DELETE CASCADE,
  FOREIGN KEY (acknowledged_by) REFERENCES operators(id) ON DELETE SET NULL,
  INDEX idx_alerts_inverter (inverter_id),
  INDEX idx_alerts_created (created_at)
);

-- Audit Logs
CREATE TABLE IF NOT EXISTS audit_logs (
  id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
  user_id VARCHAR(36) NOT NULL,
  user_role ENUM('admin', 'operator') NOT NULL,
  action VARCHAR(100) NOT NULL,
  details JSON NULL,
  ip_address VARCHAR(50) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audit_user (user_id),
  INDEX idx_audit_created (created_at)
);

-- Settings (single row)
CREATE TABLE IF NOT EXISTS settings (
  id INT PRIMARY KEY DEFAULT 1,
  stale_timeout_minutes INT DEFAULT 10,
  thresholds JSON NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert default settings row
INSERT IGNORE INTO settings (id, stale_timeout_minutes, thresholds)
VALUES (1, 10, '{"A":[90,100],"B":[75,89],"C":[50,74],"D":[25,49],"E":[0,24]}');
