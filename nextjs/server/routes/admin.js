const express = require('express');
const bcrypt = require('bcryptjs');
const { z } = require('zod');
const pool = require('../db/connection');
const { authenticate, requireAdmin, auditLog } = require('../middleware/auth');

const router = express.Router();
router.use(authenticate, requireAdmin);

// ── GET /api/admin/dashboard ──
router.get('/dashboard', async (req, res) => {
    try {
        const [[plantStats]] = await pool.query(
            `SELECT COUNT(*) AS total_plants,
              SUM(status = 'active') AS active_plants
       FROM plants WHERE status != 'decommissioned'`
        );
        const [[blockStats]] = await pool.query(`SELECT COUNT(*) AS total_blocks FROM blocks`);
        const [[invStats]] = await pool.query(
            `SELECT COUNT(*) AS total_inverters,
              SUM(current_category IN ('D','E')) AS fault_count,
              SUM(current_category = 'offline' OR is_online = 0) AS offline_count
       FROM inverters`
        );
        
        // Category breakdown for inverter status
        const [categoryStats] = await pool.query(
            `SELECT 
                current_category,
                COUNT(*) as count
             FROM inverters
             WHERE current_category IS NOT NULL
             GROUP BY current_category`
        );
        
        const categoryBreakdown = {
            A: 0, B: 0, C: 0, D: 0, E: 0, offline: 0
        };
        categoryStats.forEach(row => {
            categoryBreakdown[row.current_category] = Number(row.count);
        });
        
        const [[opStats]] = await pool.query(`SELECT COUNT(*) AS total_operators FROM operators WHERE is_active = 1`);
        const [[lastReading]] = await pool.query(`SELECT MAX(timestamp) AS last_data FROM inverter_readings`);

        res.json({
            totalPlants: Number(plantStats.total_plants),
            activePlants: Number(plantStats.active_plants),
            totalBlocks: Number(blockStats.total_blocks),
            totalInverters: Number(invStats.total_inverters),
            faultCount: Number(invStats.fault_count) || 0,
            offlineCount: Number(invStats.offline_count) || 0,
            totalOperators: Number(opStats.total_operators),
            lastDataAt: lastReading.last_data,
            categoryBreakdown,
        });
    } catch (err) {
        console.error('Admin dashboard error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── GET /api/admin/green-analytics ──
router.get('/green-analytics', async (req, res) => {
    try {
        const EMISSION_FACTOR = 0.000716;   // tonnes CO2 per kWh (India CEA 2023)
        const CARBON_PRICE_USD = 22;        // USD per tonne (voluntary carbon market)
        const TARIFF_INR = 4;               // INR per kWh (typical Indian solar PPA)
        const READING_INTERVAL_HRS = 5 / 3600; // simulator fires every 5 seconds

        // ── 1. Total energy generated in last 30 days ──
        const [[energyRow]] = await pool.query(
            `SELECT COALESCE(SUM(GREATEST(ac_power, 0) * ?), 0) AS energy_kwh,
                    COUNT(*) AS reading_count
             FROM inverter_readings
             WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)`,
            [READING_INTERVAL_HRS]
        );
        const energyKwh30d = Number(energyRow.energy_kwh) || 0;

        // ── 2. Installed capacity for PR / CUF ──
        const [[capRow]] = await pool.query(
            `SELECT COALESCE(SUM(capacity_kw), 0) AS total_kw FROM inverters WHERE is_online = 1`
        );
        const installedKw = Number(capRow.total_kw) || 1;

        // Hours in 30-day window
        const windowHrs = 30 * 24;
        const prPercent = Math.min(((energyKwh30d / (installedKw * windowHrs)) * 100), 100);
        const cufPercent = Math.min(((energyKwh30d / (installedKw * 8760)) * 100 * 12), 100); // annualise

        // ── 3. CO2 & carbon credits ──
        const co2Tonnes = energyKwh30d * EMISSION_FACTOR;
        const carbonCredits = co2Tonnes;
        const creditValueUsd = co2Tonnes * CARBON_PRICE_USD;

        // ── 4. Revenue lost from current fault inverters (Cat D / E) ──
        const [faultInverters] = await pool.query(
            `SELECT i.id, i.name, i.capacity_kw,
                    MIN(r.timestamp) AS fault_since
             FROM inverters i
             JOIN inverter_readings r ON r.inverter_id = i.id
             WHERE i.current_category IN ('D', 'E')
               AND r.category IN ('D', 'E')
               AND r.timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
             GROUP BY i.id, i.name, i.capacity_kw`
        );

        let totalLossInr = 0;
        const revenueLostBreakdown = faultInverters.map(inv => {
            const hoursInFault = inv.fault_since
                ? Math.min((Date.now() - new Date(inv.fault_since).getTime()) / 3_600_000, 168)
                : 1;
            const lossInr = Number(inv.capacity_kw) * hoursInFault * TARIFF_INR;
            totalLossInr += lossInr;
            return {
                inverter: inv.name,
                hours: Math.round(hoursInFault * 10) / 10,
                loss_inr: Math.round(lossInr),
            };
        });

        // ── 5. Impact equivalents ──
        const trees = Math.round(co2Tonnes * 45.8);       // 1 tree absorbs ~21.8 kg CO2/yr
        const carsOffRoad = Math.round(co2Tonnes / 4.6);  // avg car emits 4.6 t CO2/yr
        const homesPowered = Math.round(energyKwh30d / (877)); // avg Indian home ~877 kWh/month

        res.json({
            energy_kwh_30d: Math.round(energyKwh30d),
            co2_avoided_tonnes: Math.round(co2Tonnes * 10) / 10,
            carbon_credits: Math.round(carbonCredits * 10) / 10,
            credit_value_usd: Math.round(creditValueUsd),
            revenue_lost_inr: Math.round(totalLossInr),
            revenue_lost_breakdown: revenueLostBreakdown,
            pr_percent: Math.round(prPercent * 10) / 10,
            cuf_percent: Math.round(cufPercent * 10) / 10,
            impact_equivalents: { trees, cars_off_road: carsOffRoad, homes_powered: homesPowered },
        });
    } catch (err) {
        console.error('Green analytics error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});



// ═══════════════════ PLANT CRUD ═══════════════════

// GET /api/admin/plants
router.get('/plants', async (req, res) => {
    try {
        const [plants] = await pool.query(
            `SELECT p.*,
              COUNT(DISTINCT b.id) AS block_count,
              COUNT(DISTINCT i.id) AS inverter_count,
              a.name AS created_by_name
       FROM plants p
       LEFT JOIN blocks b ON b.plant_id = p.id
       LEFT JOIN inverters i ON i.block_id = b.id
       LEFT JOIN admins a ON a.id = p.created_by
       GROUP BY p.id
       ORDER BY p.created_at DESC`
        );
        res.json(plants);
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /api/admin/plants
router.post('/plants', async (req, res) => {
    const schema = z.object({
        name: z.string().min(1),
        location: z.string().min(1),
        status: z.enum(['active', 'maintenance', 'decommissioned']).default('active'),
        total_capacity_kw: z.number().optional().default(0),
    });
    try {
        const data = schema.parse(req.body);
        const [result] = await pool.query(
            'INSERT INTO plants (id, name, location, status, total_capacity_kw, created_by) VALUES (UUID(), ?, ?, ?, ?, ?)',
            [data.name, data.location, data.status, data.total_capacity_kw, req.user.id]
        );
        const [[plant]] = await pool.query('SELECT * FROM plants ORDER BY created_at DESC LIMIT 1');
        await auditLog(req.user.id, 'admin', 'create_plant', { plant_name: data.name }, req.ip);
        res.status(201).json(plant);
    } catch (err) {
        if (err instanceof z.ZodError) return res.status(400).json({ error: err.errors[0].message });
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/plants/:id
router.put('/plants/:id', async (req, res) => {
    const schema = z.object({
        name: z.string().min(1).optional(),
        location: z.string().min(1).optional(),
        status: z.enum(['active', 'maintenance', 'decommissioned']).optional(),
        total_capacity_kw: z.number().optional(),
    });
    try {
        const data = schema.parse(req.body);
        const fields = Object.keys(data).filter(k => data[k] !== undefined);
        if (!fields.length) return res.status(400).json({ error: 'No fields to update' });

        const setClauses = fields.map(f => `${f} = ?`).join(', ');
        const values = fields.map(f => data[f]);
        await pool.query(`UPDATE plants SET ${setClauses} WHERE id = ?`, [...values, req.params.id]);
        await auditLog(req.user.id, 'admin', 'update_plant', { plant_id: req.params.id, ...data }, req.ip);
        const [[plant]] = await pool.query('SELECT * FROM plants WHERE id = ?', [req.params.id]);
        res.json(plant);
    } catch (err) {
        if (err instanceof z.ZodError) return res.status(400).json({ error: err.errors[0].message });
        res.status(500).json({ error: 'Internal server error' });
    }
});

// DELETE /api/admin/plants/:id (soft decommission)
router.delete('/plants/:id', async (req, res) => {
    try {
        await pool.query(`UPDATE plants SET status = 'decommissioned' WHERE id = ?`, [req.params.id]);
        await auditLog(req.user.id, 'admin', 'decommission_plant', { plant_id: req.params.id }, req.ip);
        res.json({ message: 'Plant decommissioned successfully' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ═══════════════════ BLOCK CRUD ═══════════════════

// GET /api/admin/plants/:plantId/blocks
router.get('/plants/:plantId/blocks', async (req, res) => {
    try {
        const [blocks] = await pool.query(
            `SELECT b.*, COUNT(i.id) AS inverter_count
       FROM blocks b LEFT JOIN inverters i ON i.block_id = b.id
       WHERE b.plant_id = ? GROUP BY b.id ORDER BY b.name`,
            [req.params.plantId]
        );
        res.json(blocks);
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /api/admin/plants/:plantId/blocks
router.post('/plants/:plantId/blocks', async (req, res) => {
    const { name } = req.body;
    if (!name) return res.status(400).json({ error: 'Block name is required' });
    try {
        await pool.query('INSERT INTO blocks (id, plant_id, name) VALUES (UUID(), ?, ?)', [req.params.plantId, name]);
        const [[block]] = await pool.query('SELECT * FROM blocks WHERE plant_id = ? ORDER BY created_at DESC LIMIT 1', [req.params.plantId]);
        await auditLog(req.user.id, 'admin', 'create_block', { plant_id: req.params.plantId, block_name: name }, req.ip);
        res.status(201).json(block);
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/blocks/:id
router.put('/blocks/:id', async (req, res) => {
    const { name } = req.body;
    if (!name) return res.status(400).json({ error: 'Block name is required' });
    try {
        await pool.query('UPDATE blocks SET name = ? WHERE id = ?', [name, req.params.id]);
        const [[block]] = await pool.query('SELECT * FROM blocks WHERE id = ?', [req.params.id]);
        res.json(block);
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// DELETE /api/admin/blocks/:id
router.delete('/blocks/:id', async (req, res) => {
    try {
        const [[check]] = await pool.query('SELECT COUNT(*) AS cnt FROM inverters WHERE block_id = ?', [req.params.id]);
        if (Number(check.cnt) > 0) {
            return res.status(400).json({ error: 'Cannot delete block with inverters. Remove all inverters first.' });
        }
        await pool.query('DELETE FROM blocks WHERE id = ?', [req.params.id]);
        await auditLog(req.user.id, 'admin', 'delete_block', { block_id: req.params.id }, req.ip);
        res.json({ message: 'Block deleted' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ═══════════════════ INVERTER CRUD ═══════════════════

// GET /api/admin/blocks/:blockId/inverters
router.get('/blocks/:blockId/inverters', async (req, res) => {
    try {
        const [inverters] = await pool.query(
            `SELECT i.*,
              r.dc_voltage, r.dc_current, r.ac_power, r.module_temp, r.ambient_temp, r.irradiation,
              r.confidence, r.is_faulty, r.fault_type, r.timestamp AS reading_at
       FROM inverters i
       LEFT JOIN (
         SELECT ir.* FROM inverter_readings ir
         INNER JOIN (
           SELECT inverter_id, MAX(timestamp) AS max_ts FROM inverter_readings GROUP BY inverter_id
         ) latest ON ir.inverter_id = latest.inverter_id AND ir.timestamp = latest.max_ts
       ) r ON r.inverter_id = i.id
       WHERE i.block_id = ? ORDER BY i.name`,
            [req.params.blockId]
        );
        res.json(inverters.map(i => ({ ...i, is_online: !!i.is_online, is_faulty: !!i.is_faulty })));
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /api/admin/blocks/:blockId/inverters
router.post('/blocks/:blockId/inverters', async (req, res) => {
    const schema = z.object({
        name: z.string().min(1),
        serial_number: z.string().optional(),
        capacity_kw: z.number().optional().default(50),
    });
    try {
        const data = schema.parse(req.body);
        await pool.query(
            'INSERT INTO inverters (id, block_id, name, serial_number, capacity_kw) VALUES (UUID(), ?, ?, ?, ?)',
            [req.params.blockId, data.name, data.serial_number || null, data.capacity_kw]
        );
        const [[inv]] = await pool.query('SELECT * FROM inverters WHERE block_id = ? ORDER BY installed_at DESC LIMIT 1', [req.params.blockId]);
        await auditLog(req.user.id, 'admin', 'add_inverter', { block_id: req.params.blockId, inverter_name: data.name }, req.ip);
        res.status(201).json(inv);
    } catch (err) {
        if (err instanceof z.ZodError) return res.status(400).json({ error: err.errors[0].message });
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/inverters/:id
router.put('/inverters/:id', async (req, res) => {
    const schema = z.object({
        name: z.string().optional(),
        serial_number: z.string().optional(),
        capacity_kw: z.number().optional(),
    });
    try {
        const data = schema.parse(req.body);
        const fields = Object.keys(data).filter(k => data[k] !== undefined);
        if (!fields.length) return res.status(400).json({ error: 'No fields to update' });
        const setClauses = fields.map(f => `${f} = ?`).join(', ');
        await pool.query(`UPDATE inverters SET ${setClauses} WHERE id = ?`, [...fields.map(f => data[f]), req.params.id]);
        const [[inv]] = await pool.query('SELECT * FROM inverters WHERE id = ?', [req.params.id]);
        res.json(inv);
    } catch (err) {
        if (err instanceof z.ZodError) return res.status(400).json({ error: err.errors[0].message });
        res.status(500).json({ error: 'Internal server error' });
    }
});

// DELETE /api/admin/inverters/:id
router.delete('/inverters/:id', async (req, res) => {
    try {
        await pool.query('DELETE FROM inverters WHERE id = ?', [req.params.id]);
        await auditLog(req.user.id, 'admin', 'delete_inverter', { inverter_id: req.params.id }, req.ip);
        res.json({ message: 'Inverter removed' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ═══════════════════ OPERATOR CRUD ═══════════════════

// GET /api/admin/operators
router.get('/operators', async (req, res) => {
    try {
        const [operators] = await pool.query(
            `SELECT o.id, o.name, o.email, o.is_active, o.created_at, o.last_login,
              GROUP_CONCAT(p.name ORDER BY p.name SEPARATOR ', ') AS assigned_plant_names,
              GROUP_CONCAT(p.id) AS assigned_plant_ids
       FROM operators o
       LEFT JOIN operator_plant_access opa ON opa.operator_id = o.id
       LEFT JOIN plants p ON opa.plant_id = p.id
       GROUP BY o.id
       ORDER BY o.created_at DESC`
        );
        res.json(operators.map(op => ({
            ...op,
            is_active: !!op.is_active,
            assignedPlantIds: op.assigned_plant_ids ? op.assigned_plant_ids.split(',') : [],
        })));
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /api/admin/operators
router.post('/operators', async (req, res) => {
    const schema = z.object({
        name: z.string().min(1),
        email: z.string().email(),
        password: z.string().min(8),
    });
    try {
        const { name, email, password } = schema.parse(req.body);

        // Check duplicate email
        const [existing] = await pool.query('SELECT id FROM operators WHERE email = ?', [email]);
        if (existing.length) return res.status(409).json({ error: 'An operator with this email already exists' });

        const hash = await bcrypt.hash(password, 12);
        const opId = require('crypto').randomUUID();
        await pool.query(
            'INSERT INTO operators (id, name, email, password_hash, created_by) VALUES (?, ?, ?, ?, ?)',
            [opId, name, email, hash, req.user.id]
        );
        await auditLog(req.user.id, 'admin', 'add_operator', { operator_name: name, email }, req.ip);
        const [[op]] = await pool.query('SELECT id, name, email, is_active, created_at FROM operators WHERE id = ?', [opId]);
        res.status(201).json({ ...op, is_active: !!op.is_active });
    } catch (err) {
        if (err instanceof z.ZodError) return res.status(400).json({ error: err.errors[0].message });
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/operators/:id
router.put('/operators/:id', async (req, res) => {
    const schema = z.object({ name: z.string().min(1).optional(), email: z.string().email().optional() });
    try {
        const data = schema.parse(req.body);
        if (data.email) {
            const [dup] = await pool.query('SELECT id FROM operators WHERE email = ? AND id != ?', [data.email, req.params.id]);
            if (dup.length) return res.status(409).json({ error: 'Email already in use' });
        }
        const fields = Object.keys(data).filter(k => data[k] !== undefined);
        if (!fields.length) return res.status(400).json({ error: 'No fields to update' });
        const setClauses = fields.map(f => `${f} = ?`).join(', ');
        await pool.query(`UPDATE operators SET ${setClauses} WHERE id = ?`, [...fields.map(f => data[f]), req.params.id]);
        await auditLog(req.user.id, 'admin', 'update_operator', { operator_id: req.params.id }, req.ip);
        const [[op]] = await pool.query('SELECT id, name, email, is_active FROM operators WHERE id = ?', [req.params.id]);
        res.json({ ...op, is_active: !!op.is_active });
    } catch (err) {
        if (err instanceof z.ZodError) return res.status(400).json({ error: err.errors[0].message });
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/operators/:id/assign-plants
router.put('/operators/:id/assign-plants', async (req, res) => {
    const { plant_ids } = req.body;
    if (!Array.isArray(plant_ids)) return res.status(400).json({ error: 'plant_ids must be an array' });
    try {
        await pool.query('DELETE FROM operator_plant_access WHERE operator_id = ?', [req.params.id]);
        if (plant_ids.length) {
            const values = plant_ids.map(pid => [req.params.id, pid]);
            await pool.query('INSERT INTO operator_plant_access (operator_id, plant_id) VALUES ?', [values]);
        }
        await auditLog(req.user.id, 'admin', 'assign_plants', { operator_id: req.params.id, plant_ids }, req.ip);
        res.json({ message: 'Plant assignments updated' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/operators/:id/deactivate
router.put('/operators/:id/deactivate', async (req, res) => {
    try {
        await pool.query('UPDATE operators SET is_active = FALSE WHERE id = ?', [req.params.id]);
        await auditLog(req.user.id, 'admin', 'deactivate_operator', { operator_id: req.params.id }, req.ip);
        res.json({ message: 'Operator deactivated' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/operators/:id/activate
router.put('/operators/:id/activate', async (req, res) => {
    try {
        await pool.query('UPDATE operators SET is_active = TRUE WHERE id = ?', [req.params.id]);
        await auditLog(req.user.id, 'admin', 'activate_operator', { operator_id: req.params.id }, req.ip);
        res.json({ message: 'Operator activated' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// DELETE /api/admin/operators/:id
router.delete('/operators/:id', async (req, res) => {
    try {
        const [[op]] = await pool.query('SELECT name FROM operators WHERE id = ?', [req.params.id]);
        await pool.query('DELETE FROM operators WHERE id = ?', [req.params.id]);
        await auditLog(req.user.id, 'admin', 'delete_operator', { operator_id: req.params.id, name: op?.name }, req.ip);
        res.json({ message: 'Operator deleted' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /api/admin/operators/:id/reset-password
router.post('/operators/:id/reset-password', async (req, res) => {
    const { new_password } = req.body;
    if (!new_password || new_password.length < 8) {
        return res.status(400).json({ error: 'new_password must be at least 8 characters' });
    }
    try {
        const hash = await bcrypt.hash(new_password, 12);
        await pool.query('UPDATE operators SET password_hash = ?, must_reset_password = TRUE WHERE id = ?', [hash, req.params.id]);
        await auditLog(req.user.id, 'admin', 'reset_operator_password', { operator_id: req.params.id }, req.ip);
        res.json({ message: 'Password reset. Operator must change on next login.' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ═══════════════════ ALERTS ═══════════════════

// GET /api/admin/alerts
router.get('/alerts', async (req, res) => {
    try {
        const { plant_id, type, acknowledged, limit = 100, offset = 0 } = req.query;
        let whereClause = 'WHERE 1=1';
        const params = [];

        if (plant_id) { whereClause += ' AND p.id = ?'; params.push(plant_id); }
        if (type) { whereClause += ' AND a.type = ?'; params.push(type); }
        if (acknowledged !== undefined) { whereClause += ' AND a.acknowledged = ?'; params.push(acknowledged === 'true' ? 1 : 0); }

        const [alerts] = await pool.query(
            `SELECT a.id, a.type, a.message, a.category_from, a.category_to, a.acknowledged, a.acknowledged_at, a.created_at,
              i.id AS inverter_id, i.name AS inverter_name,
              b.id AS block_id, b.name AS block_name,
              p.id AS plant_id, p.name AS plant_name,
              op.name AS acknowledged_by_name
       FROM alerts a
       JOIN inverters i ON a.inverter_id = i.id
       JOIN blocks b ON i.block_id = b.id
       JOIN plants p ON b.plant_id = p.id
       LEFT JOIN operators op ON a.acknowledged_by = op.id
       ${whereClause}
       ORDER BY a.created_at DESC
       LIMIT ? OFFSET ?`,
            [...params, Number(limit), Number(offset)]
        );
        res.json(alerts.map(a => ({ ...a, acknowledged: !!a.acknowledged })));
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ═══════════════════ AUDIT LOGS ═══════════════════

// GET /api/admin/audit-logs
router.get('/audit-logs', async (req, res) => {
    try {
        const { user_id, action, limit = 50, offset = 0 } = req.query;
        let whereClause = 'WHERE 1=1';
        const params = [];

        if (user_id) { whereClause += ' AND user_id = ?'; params.push(user_id); }
        if (action) { whereClause += ' AND action LIKE ?'; params.push(`%${action}%`); }

        const [logs] = await pool.query(
            `SELECT al.*,
              COALESCE(a.name, op.name) AS user_name
       FROM audit_logs al
       LEFT JOIN admins a ON al.user_id = a.id AND al.user_role = 'admin'
       LEFT JOIN operators op ON al.user_id = op.id AND al.user_role = 'operator'
       ${whereClause}
       ORDER BY al.created_at DESC
       LIMIT ? OFFSET ?`,
            [...params, Number(limit), Number(offset)]
        );

        res.json(logs.map(l => ({
            ...l,
            details: l.details ? (typeof l.details === 'string' ? JSON.parse(l.details) : l.details) : {},
        })));
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ═══════════════════ SETTINGS ═══════════════════

// GET /api/admin/settings
router.get('/settings', async (req, res) => {
    try {
        const [[settings]] = await pool.query('SELECT * FROM settings WHERE id = 1');
        res.json({
            staleTimeoutMinutes: settings.stale_timeout_minutes,
            thresholds: settings.thresholds
                ? (typeof settings.thresholds === 'string' ? JSON.parse(settings.thresholds) : settings.thresholds)
                : {},
        });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

// PUT /api/admin/settings
router.put('/settings', async (req, res) => {
    try {
        const { staleTimeoutMinutes, thresholds } = req.body;
        await pool.query(
            'UPDATE settings SET stale_timeout_minutes = ?, thresholds = ? WHERE id = 1',
            [staleTimeoutMinutes, JSON.stringify(thresholds)]
        );
        await auditLog(req.user.id, 'admin', 'update_settings', { staleTimeoutMinutes }, req.ip);
        res.json({ message: 'Settings updated' });
    } catch (err) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
