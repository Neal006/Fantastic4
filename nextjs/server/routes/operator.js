const express = require('express');
const { z } = require('zod');
const pool = require('../db/connection');
const { authenticate, auditLog } = require('../middleware/auth');

const router = express.Router();
router.use(authenticate);

// Helper: verify operator can access this plant
async function canAccessPlant(operatorId, plantId) {
    const [rows] = await pool.query(
        'SELECT 1 FROM operator_plant_access WHERE operator_id = ? AND plant_id = ?',
        [operatorId, plantId]
    );
    return rows.length > 0;
}

// Helper: get plant_id from block_id
async function getPlantIdFromBlock(blockId) {
    const [rows] = await pool.query('SELECT plant_id FROM blocks WHERE id = ?', [blockId]);
    return rows[0]?.plant_id || null;
}

// Helper: get plant_id from inverter_id
async function getPlantIdFromInverter(inverterId) {
    const [rows] = await pool.query(
        'SELECT b.plant_id FROM inverters i JOIN blocks b ON i.block_id = b.id WHERE i.id = ?',
        [inverterId]
    );
    return rows[0]?.plant_id || null;
}

// ── GET /api/operator/dashboard ──
router.get('/dashboard', async (req, res) => {
    try {
        const op = req.user;
        const assignedPlants = op.assignedPlants || [];
        if (!assignedPlants.length) {
            return res.json({ totalInverters: 0, healthy: 0, warning: 0, critical: 0, offline: 0, needsAttention: [], recentAlerts: [] });
        }

        const placeholders = assignedPlants.map(() => '?').join(',');

        // Inverter stats
        const [stats] = await pool.query(
            `SELECT
        COUNT(*) AS total,
        SUM(current_category IN ('A','B')) AS healthy,
        SUM(current_category = 'C') AS warning,
        SUM(current_category IN ('D','E')) AS critical,
        SUM(current_category = 'offline' OR is_online = 0) AS offline
       FROM inverters i
       JOIN blocks b ON i.block_id = b.id
       WHERE b.plant_id IN (${placeholders})`,
            assignedPlants
        );

        // Needs attention inverters (D and E)
        const [needsAttention] = await pool.query(
            `SELECT i.id, i.name, i.current_category, i.last_data_at, b.id AS block_id, b.name AS block_name, p.id AS plant_id, p.name AS plant_name
       FROM inverters i
       JOIN blocks b ON i.block_id = b.id
       JOIN plants p ON b.plant_id = p.id
       WHERE b.plant_id IN (${placeholders}) AND i.current_category IN ('D','E')
       ORDER BY i.current_category DESC
       LIMIT 20`,
            assignedPlants
        );

        const pla2 = assignedPlants.map(() => '?').join(',');
        // Recent alerts
        const [recentAlerts] = await pool.query(
            `SELECT a.id, a.type, a.message, a.acknowledged, a.created_at,
              i.name AS inverter_name, b.plant_id
       FROM alerts a
       JOIN inverters i ON a.inverter_id = i.id
       JOIN blocks b ON i.block_id = b.id
       WHERE b.plant_id IN (${pla2})
       ORDER BY a.created_at DESC
       LIMIT 10`,
            assignedPlants
        );

        res.json({
            totalInverters: Number(stats[0].total) || 0,
            healthy: Number(stats[0].healthy) || 0,
            warning: Number(stats[0].warning) || 0,
            critical: Number(stats[0].critical) || 0,
            offline: Number(stats[0].offline) || 0,
            needsAttention,
            recentAlerts,
        });
    } catch (err) {
        console.error('Operator dashboard error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/operator/plants ──
router.get('/plants', async (req, res) => {
    try {
        const assignedPlants = req.user.assignedPlants || [];
        if (!assignedPlants.length) return res.json([]);

        const placeholders = assignedPlants.map(() => '?').join(',');
        const [plants] = await pool.query(
            `SELECT p.id, p.name, p.location, p.status,
              b.id AS block_id, b.name AS block_name,
              COUNT(i.id) AS inverter_count
       FROM plants p
       LEFT JOIN blocks b ON b.plant_id = p.id
       LEFT JOIN inverters i ON i.block_id = b.id
       WHERE p.id IN (${placeholders}) AND p.status != 'decommissioned'
       GROUP BY p.id, p.name, p.location, p.status, b.id, b.name
       ORDER BY p.name, b.name`,
            assignedPlants
        );

        // Group into plants -> blocks structure
        const plantMap = new Map();
        for (const row of plants) {
            if (!plantMap.has(row.id)) {
                plantMap.set(row.id, { id: row.id, name: row.name, location: row.location, status: row.status, blocks: [] });
            }
            if (row.block_id) {
                plantMap.get(row.id).blocks.push({
                    id: row.block_id,
                    name: row.block_name,
                    inverterCount: Number(row.inverter_count),
                });
            }
        }

        res.json(Array.from(plantMap.values()));
    } catch (err) {
        console.error('Plants error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/operator/plants/:plantId/blocks/:blockId/inverters ──
router.get('/plants/:plantId/blocks/:blockId/inverters', async (req, res) => {
    try {
        const { plantId, blockId } = req.params;
        if (req.user.role !== 'admin') {
            const canAccess = await canAccessPlant(req.user.id, plantId);
            if (!canAccess) return res.status(403).json({ error: 'Access denied to this plant' });
        }

        const [inverters] = await pool.query(
            `SELECT i.id, i.name, i.capacity_kw, i.current_category, i.is_online, i.last_data_at,
              r.dc_voltage, r.dc_current, r.ac_power, r.module_temp, r.ambient_temp, r.irradiation,
              r.confidence, r.is_faulty, r.fault_type, r.shap_values, r.timestamp AS reading_at
       FROM inverters i
       LEFT JOIN (
         SELECT *
         FROM (
           SELECT ir.*,
                  ROW_NUMBER() OVER (PARTITION BY ir.inverter_id ORDER BY ir.timestamp DESC) AS rn
           FROM inverter_readings ir
         ) ranked
         WHERE ranked.rn = 1
       ) r ON r.inverter_id = i.id
       WHERE i.block_id = ?
       ORDER BY i.name`,
            [blockId]
        );

        const data = inverters.map(inv => ({
            ...inv,
            shap_values: inv.shap_values ? (typeof inv.shap_values === 'string' ? JSON.parse(inv.shap_values) : inv.shap_values) : [],
            is_online: !!inv.is_online,
            is_faulty: !!inv.is_faulty,
        }));

        res.json(data);
    } catch (err) {
        console.error('Inverter grid error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/operator/inverters/:id ──
router.get('/inverters/:id', async (req, res) => {
    try {
        const [rows] = await pool.query(
            `SELECT i.*, b.name AS block_name, b.plant_id, p.name AS plant_name
       FROM inverters i
       JOIN blocks b ON i.block_id = b.id
       JOIN plants p ON b.plant_id = p.id
       WHERE i.id = ?`,
            [req.params.id]
        );
        if (!rows.length) return res.status(404).json({ error: 'Inverter not found' });

        const inv = rows[0];
        if (req.user.role !== 'admin') {
            const canAccess = await canAccessPlant(req.user.id, inv.plant_id);
            if (!canAccess) return res.status(403).json({ error: 'Access denied' });
        }

        // Latest reading
        const [readings] = await pool.query(
            `SELECT * FROM inverter_readings WHERE inverter_id = ? ORDER BY timestamp DESC LIMIT 1`,
            [req.params.id]
        );
        const latestReading = readings[0] || null;

        res.json({
            ...inv,
            is_online: !!inv.is_online,
            latestReading: latestReading ? {
                ...latestReading,
                shap_values: latestReading.shap_values
                    ? (typeof latestReading.shap_values === 'string' ? JSON.parse(latestReading.shap_values) : latestReading.shap_values)
                    : [],
            } : null,
        });
    } catch (err) {
        console.error('Inverter detail error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/operator/inverters/:id/readings ──
router.get('/inverters/:id/readings', async (req, res) => {
    try {
        const range = req.query.range || '24h';
        const hours = range === '24h' ? 24 : range === '48h' ? 48 : 24;

        // Access check
        if (req.user.role !== 'admin') {
            const plantId = await getPlantIdFromInverter(req.params.id);
            if (!plantId || !(await canAccessPlant(req.user.id, plantId))) {
                return res.status(403).json({ error: 'Access denied' });
            }
        }

        const [readings] = await pool.query(
            `SELECT timestamp, dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, category, confidence
       FROM inverter_readings
       WHERE inverter_id = ? AND timestamp >= DATE_SUB(NOW(), INTERVAL ? HOUR)
       ORDER BY timestamp ASC`,
            [req.params.id, hours]
        );

        res.json(readings);
    } catch (err) {
        console.error('Readings error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/operator/inverters/:id/faults ──
router.get('/inverters/:id/faults', async (req, res) => {
    try {
        if (req.user.role !== 'admin') {
            const plantId = await getPlantIdFromInverter(req.params.id);
            if (!plantId || !(await canAccessPlant(req.user.id, plantId))) {
                return res.status(403).json({ error: 'Access denied' });
            }
        }

        const [faults] = await pool.query(
            `SELECT id, timestamp, category, fault_type
       FROM inverter_readings
       WHERE inverter_id = ? AND is_faulty = TRUE
       ORDER BY timestamp DESC
       LIMIT 50`,
            [req.params.id]
        );

        res.json(faults);
    } catch (err) {
        console.error('Fault history error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/operator/alerts ──
router.get('/alerts', async (req, res) => {
    try {
        const assignedPlants = req.user.assignedPlants || [];
        if (!assignedPlants.length) return res.json([]);

        const placeholders = assignedPlants.map(() => '?').join(',');
        const [alerts] = await pool.query(
            `SELECT a.id, a.type, a.message, a.category_from, a.category_to, a.acknowledged, a.acknowledged_at, a.created_at,
              i.id AS inverter_id, i.name AS inverter_name, b.id AS block_id, b.name AS block_name,
              p.id AS plant_id, p.name AS plant_name,
              op.name AS acknowledged_by_name
       FROM alerts a
       JOIN inverters i ON a.inverter_id = i.id
       JOIN blocks b ON i.block_id = b.id
       JOIN plants p ON b.plant_id = p.id
       LEFT JOIN operators op ON a.acknowledged_by = op.id
       WHERE b.plant_id IN (${placeholders})
       ORDER BY a.created_at DESC
       LIMIT 100`,
            assignedPlants
        );

        res.json(alerts.map(a => ({ ...a, acknowledged: !!a.acknowledged })));
    } catch (err) {
        console.error('Alerts error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── POST /api/operator/alerts/:id/acknowledge ──
router.post('/alerts/:id/acknowledge', async (req, res) => {
    try {
        const [alerts] = await pool.query('SELECT * FROM alerts WHERE id = ?', [req.params.id]);
        if (!alerts.length) return res.status(404).json({ error: 'Alert not found' });

        const alert = alerts[0];

        // Access check
        if (req.user.role !== 'admin') {
            const plantId = await getPlantIdFromInverter(alert.inverter_id);
            if (!plantId || !(await canAccessPlant(req.user.id, plantId))) {
                return res.status(403).json({ error: 'Access denied' });
            }
        }

        await pool.query(
            'UPDATE alerts SET acknowledged = TRUE, acknowledged_by = ?, acknowledged_at = NOW() WHERE id = ?',
            [req.user.id, req.params.id]
        );

        await auditLog(req.user.id, req.user.role, 'acknowledge_alert', { alert_id: req.params.id }, req.ip);

        res.json({ message: 'Alert acknowledged' });
    } catch (err) {
        console.error('Acknowledge error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
