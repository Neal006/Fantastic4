const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { z } = require('zod');
const pool = require('../db/connection');
const { auditLog } = require('../middleware/auth');

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || 'solarwatch_secret';
const JWT_EXPIRES = '8h';

// ── POST /api/auth/login ──
const loginSchema = z.object({
    id: z.string().min(1, 'User ID is required'),
    password: z.string().min(1, 'Password is required'),
});

router.post('/login', async (req, res) => {
    try {
        const { id, password } = loginSchema.parse(req.body);
        const ip = req.ip || req.headers['x-forwarded-for'] || 'unknown';

        // Try admin first
        let [rows] = await pool.query(
            'SELECT id, name, email, password_hash, "admin" AS role, NULL AS is_active, NULL AS assigned_plants FROM admins WHERE id = ? OR email = ?',
            [id, id]
        );

        // If not admin, try operator
        if (!rows.length) {
            [rows] = await pool.query(
                `SELECT o.id, o.name, o.email, o.password_hash, 'operator' AS role, o.is_active, o.must_reset_password,
         GROUP_CONCAT(opa.plant_id) AS assigned_plants
         FROM operators o
         LEFT JOIN operator_plant_access opa ON opa.operator_id = o.id
         WHERE o.id = ? OR o.email = ?
         GROUP BY o.id`,
                [id, id]
            );
        }

        if (!rows.length) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const user = rows[0];

        // Check operator is active
        if (user.role === 'operator' && !user.is_active) {
            return res.status(403).json({ error: 'Your account has been deactivated. Contact your administrator.' });
        }

        const valid = await bcrypt.compare(password, user.password_hash);
        if (!valid) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const assignedPlants = user.assigned_plants
            ? user.assigned_plants.split(',').filter(Boolean)
            : [];

        const payload = {
            id: user.id,
            name: user.name,
            email: user.email,
            role: user.role,
            assignedPlants,
        };

        const token = jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES });

        // Update last_login for operators
        if (user.role === 'operator') {
            await pool.query('UPDATE operators SET last_login = NOW() WHERE id = ?', [user.id]);
        }

        // Set httpOnly cookie
        res.cookie('sw_token', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'lax',
            maxAge: 8 * 60 * 60 * 1000, // 8 hours
        });

        res.json({
            token,
            user: {
                id: user.id,
                name: user.name,
                email: user.email,
                role: user.role,
                assignedPlants,
                mustResetPassword: !!user.must_reset_password,
            },
        });

    } catch (err) {
        if (err instanceof z.ZodError) {
            return res.status(400).json({ error: err.errors[0].message });
        }
        console.error('Login error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── POST /api/auth/logout ──
router.post('/logout', (req, res) => {
    res.clearCookie('sw_token');
    res.json({ message: 'Logged out successfully' });
});

// ── POST /api/auth/reset-password ──
router.post('/reset-password', async (req, res) => {
    const { authenticate } = require('../middleware/auth');
    // Use inline auth since we need it here
    let token = null;
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) token = authHeader.slice(7);
    if (!token && req.cookies?.sw_token) token = req.cookies.sw_token;
    if (!token) return res.status(401).json({ error: 'Authentication required' });

    let user;
    try {
        user = jwt.verify(token, JWT_SECRET);
    } catch {
        return res.status(401).json({ error: 'Invalid token' });
    }

    const schema = z.object({
        currentPassword: z.string().min(1),
        newPassword: z.string().min(8, 'Password must be at least 8 characters'),
        confirmPassword: z.string(),
    }).refine(d => d.newPassword === d.confirmPassword, { message: 'Passwords do not match' });

    try {
        const { currentPassword, newPassword } = schema.parse(req.body);

        const table = user.role === 'admin' ? 'admins' : 'operators';
        const [rows] = await pool.query(`SELECT password_hash FROM ${table} WHERE id = ?`, [user.id]);
        if (!rows.length) return res.status(404).json({ error: 'User not found' });

        const valid = await bcrypt.compare(currentPassword, rows[0].password_hash);
        if (!valid) return res.status(401).json({ error: 'Current password is incorrect' });

        const hash = await bcrypt.hash(newPassword, 12);
        await pool.query(`UPDATE ${table} SET password_hash = ? WHERE id = ?`, [hash, user.id]);

        if (user.role === 'operator') {
            await pool.query('UPDATE operators SET must_reset_password = FALSE WHERE id = ?', [user.id]);
        }

        res.json({ message: 'Password updated successfully' });
    } catch (err) {
        if (err instanceof z.ZodError) {
            return res.status(400).json({ error: err.errors[0].message });
        }
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ── POST /api/auth/force-reset (admin only) ──
router.post('/force-reset', async (req, res) => {
    let token = null;
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) token = authHeader.slice(7);
    if (!token && req.cookies?.sw_token) token = req.cookies.sw_token;
    if (!token) return res.status(401).json({ error: 'Authentication required' });

    let user;
    try {
        user = jwt.verify(token, JWT_SECRET);
    } catch {
        return res.status(401).json({ error: 'Invalid token' });
    }
    if (user.role !== 'admin') return res.status(403).json({ error: 'Admin access required' });

    const { operator_id, new_password } = req.body;
    if (!operator_id || !new_password || new_password.length < 8) {
        return res.status(400).json({ error: 'operator_id and new_password (min 8 chars) required' });
    }

    const hash = await bcrypt.hash(new_password, 12);
    await pool.query(
        'UPDATE operators SET password_hash = ?, must_reset_password = TRUE WHERE id = ?',
        [hash, operator_id]
    );

    await auditLog(user.id, 'admin', 'force_reset_password', { operator_id }, req.ip);
    res.json({ message: 'Password reset successfully. Operator must change on next login.' });
});

module.exports = router;
