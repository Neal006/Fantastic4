-- ============================================================
-- Lumin AI — New Database Schema
-- Database: Hackamined_new
-- ============================================================
-- Run this to create the database + all tables from scratch.
-- Schema updated to match the ML model JSON output format,
-- including enriched SHAP structure and a chat_logs table.
-- ============================================================

CREATE DATABASE IF NOT EXISTS Hackamined_new
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE Hackamined_new;

-- ── Admins ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS admins (
  id            VARCHAR(36)   NOT NULL PRIMARY KEY DEFAULT (UUID()),
  name          VARCHAR(100)  NOT NULL,
  email         VARCHAR(100)  NOT NULL UNIQUE,
  password_hash VARCHAR(255)  NOT NULL,
  created_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
  updated_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- ── Operators ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS operators (
  id                  VARCHAR(36)   NOT NULL PRIMARY KEY DEFAULT (UUID()),
  name                VARCHAR(100)  NOT NULL,
  email               VARCHAR(100)  NOT NULL UNIQUE,
  password_hash       VARCHAR(255)  NOT NULL,
  is_active           TINYINT(1)    DEFAULT 1,
  must_reset_password TINYINT(1)    DEFAULT 0,
  created_by          VARCHAR(36)   NULL,
  created_at          TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
  last_login          TIMESTAMP     NULL,
  FOREIGN KEY (created_by) REFERENCES admins(id) ON DELETE SET NULL
);

-- ── Plants ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plants (
  id               VARCHAR(36)   NOT NULL PRIMARY KEY DEFAULT (UUID()),
  name             VARCHAR(150)  NOT NULL,
  location         VARCHAR(200)  NOT NULL,
  status           ENUM('active','maintenance','decommissioned') DEFAULT 'active',
  total_capacity_kw DECIMAL(10,2) DEFAULT 0.00,
  created_by       VARCHAR(36)   NULL,
  created_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
  updated_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES admins(id) ON DELETE SET NULL
);

-- ── Operator ↔ Plant Access ────────────────────────────────
CREATE TABLE IF NOT EXISTS operator_plant_access (
  operator_id VARCHAR(36) NOT NULL,
  plant_id    VARCHAR(36) NOT NULL,
  assigned_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (operator_id, plant_id),
  FOREIGN KEY (operator_id) REFERENCES operators(id) ON DELETE CASCADE,
  FOREIGN KEY (plant_id)    REFERENCES plants(id)    ON DELETE CASCADE
);

-- ── Blocks ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS blocks (
  id         VARCHAR(36)  NOT NULL PRIMARY KEY DEFAULT (UUID()),
  plant_id   VARCHAR(36)  NOT NULL,
  name       VARCHAR(100) NOT NULL,
  created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
);

-- ── Inverters ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS inverters (
  id               VARCHAR(36)   NOT NULL PRIMARY KEY DEFAULT (UUID()),
  block_id         VARCHAR(36)   NOT NULL,
  name             VARCHAR(100)  NOT NULL,
  serial_number    VARCHAR(100)  NULL,
  capacity_kw      DECIMAL(8,2)  DEFAULT 50.00,
  current_category ENUM('A','B','C','D','E','offline') DEFAULT 'A',
  is_online        TINYINT(1)    DEFAULT 1,
  last_data_at     TIMESTAMP     NULL,
  installed_at     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
);

-- ── Inverter Readings ─────────────────────────────────────
-- shap_values column stores the full ML model shap output:
--   [{"feature": "power_rmean_24h", "value": 0.342}, ...]
-- probabilities column stores the ML model probability map:
--   {"no_risk": 0.95, "degradation_risk": 0.03, "shutdown_risk": 0.02}
CREATE TABLE IF NOT EXISTS inverter_readings (
  id            VARCHAR(36)   NOT NULL PRIMARY KEY DEFAULT (UUID()),
  inverter_id   VARCHAR(36)   NOT NULL,
  timestamp     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  -- ML model outputs
  category      ENUM('A','B','C','D','E','offline') NOT NULL,
  confidence    DECIMAL(5,4)  DEFAULT 0.9000,
  probabilities JSON          NULL,       -- {no_risk, degradation_risk, shutdown_risk}
  is_faulty     TINYINT(1)    DEFAULT 0,
  fault_type    VARCHAR(100)  NULL,
  -- Sensor readings (raw telemetry)
  dc_voltage    DECIMAL(8,2)  DEFAULT 0.00,
  dc_current    DECIMAL(8,4)  DEFAULT 0.0000,
  ac_power      DECIMAL(10,3) DEFAULT 0.000,
  module_temp   DECIMAL(6,2)  DEFAULT 0.00,
  ambient_temp  DECIMAL(6,2)  DEFAULT 0.00,
  irradiation   DECIMAL(8,2)  DEFAULT 0.00,
  -- SHAP explainability (ML model format: [{feature, value}])
  shap_values   JSON          NULL,
  FOREIGN KEY (inverter_id) REFERENCES inverters(id) ON DELETE CASCADE,
  INDEX idx_inverter_timestamp (inverter_id, timestamp)
);

-- ── Alerts ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
  id               VARCHAR(36)   NOT NULL PRIMARY KEY DEFAULT (UUID()),
  inverter_id      VARCHAR(36)   NOT NULL,
  type             ENUM('warning','critical','info') NOT NULL,
  message          TEXT          NOT NULL,
  category_from    ENUM('A','B','C','D','E','offline') NULL,
  category_to      ENUM('A','B','C','D','E','offline') NULL,
  acknowledged     TINYINT(1)    DEFAULT 0,
  acknowledged_by  VARCHAR(36)   NULL,
  acknowledged_at  TIMESTAMP     NULL,
  created_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (inverter_id)     REFERENCES inverters(id)  ON DELETE CASCADE,
  FOREIGN KEY (acknowledged_by) REFERENCES operators(id)  ON DELETE SET NULL,
  INDEX idx_alerts_inverter (inverter_id),
  INDEX idx_alerts_created  (created_at)
);

-- ── Audit Logs ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
  id         VARCHAR(36)  NOT NULL PRIMARY KEY DEFAULT (UUID()),
  user_id    VARCHAR(36)  NOT NULL,
  user_role  ENUM('admin','operator') NOT NULL,
  action     VARCHAR(100) NOT NULL,
  details    JSON         NULL,
  ip_address VARCHAR(50)  NULL,
  created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audit_user    (user_id),
  INDEX idx_audit_created (created_at)
);

-- ── Chat Logs ─────────────────────────────────────────────
-- Stores all AI chatbot conversations per user.
-- user_id may be NULL for unauthenticated/public queries.
-- role: 'user' for messages from the operator/admin,
--       'assistant' for GenAI responses
CREATE TABLE IF NOT EXISTS chat_logs (
  id           VARCHAR(36)  NOT NULL PRIMARY KEY DEFAULT (UUID()),
  session_id   VARCHAR(36)  NOT NULL,             -- groups a conversation thread
  user_id      VARCHAR(36)  NULL,                 -- FK to admins or operators (see user_role)
  user_role    ENUM('admin','operator','public') DEFAULT 'public',
  role         ENUM('user','assistant') NOT NULL, -- which side of the conversation
  message      TEXT         NOT NULL,             -- the message content
  inverter_id  VARCHAR(36)  NULL,                 -- optional: context inverter
  metadata     JSON         NULL,                 -- e.g. sources_used, tokens, model_version
  created_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (inverter_id) REFERENCES inverters(id) ON DELETE SET NULL,
  INDEX idx_chat_session    (session_id),
  INDEX idx_chat_user       (user_id),
  INDEX idx_chat_created    (created_at)
);

-- ── Settings ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS settings (
  id                    INT  NOT NULL PRIMARY KEY DEFAULT 1,
  stale_timeout_minutes INT  DEFAULT 10,
  thresholds            JSON NULL,
  updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- ── Default seed data ─────────────────────────────────────
INSERT IGNORE INTO settings (id, stale_timeout_minutes, thresholds)
VALUES (1, 10, '{"A":[90,100],"B":[75,89],"C":[50,74],"D":[25,49],"E":[0,24]}');
