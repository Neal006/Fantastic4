const jwt = require('jsonwebtoken');
const pool = require('../db/connection');

const JWT_SECRET = process.env.JWT_SECRET || 'solarwatch_secret';

/**
 * Authenticate middleware — verifies JWT token from Authorization header or cookie
 */
async function authenticate(req, res, next) {
    try {
        let token = null;

        // Check Authorization header first
        const authHeader = req.headers.authorization;
        if (authHeader && authHeader.startsWith('Bearer ')) {
            token = authHeader.slice(7);
        }

        // Fall back to cookie
        if (!token && req.cookies && req.cookies.sw_token) {
            token = req.cookies.sw_token;
        }

        if (!token) {
            return res.status(401).json({ error: 'No authentication token provided' });
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        req.user = decoded;

        // Check if operator is still active
        if (decoded.role === 'operator') {
            const [rows] = await pool.query(
                'SELECT is_active FROM operators WHERE id = ?',
                [decoded.id]
            );
            if (!rows.length || !rows[0].is_active) {
                return res.status(401).json({ error: 'Account deactivated. Please contact your administrator.' });
            }
        }

        next();
    } catch (err) {
        if (err.name === 'TokenExpiredError') {
            return res.status(401).json({ error: 'Session expired. Please log in again.' });
        }
        return res.status(401).json({ error: 'Invalid token' });
    }
}

/**
 * Admin-only middleware
 */
function requireAdmin(req, res, next) {
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
    }
    next();
}

/**
 * Operator or Admin middleware
 */
function requireOperatorOrAdmin(req, res, next) {
    if (!req.user || !['operator', 'admin'].includes(req.user.role)) {
        return res.status(403).json({ error: 'Operator or admin access required' });
    }
    next();
}

/**
 * Log audit action helper
 */
async function auditLog(userId, userRole, action, details, ipAddress) {
    try {
        await pool.query(
            'INSERT INTO audit_logs (id, user_id, user_role, action, details, ip_address) VALUES (UUID(), ?, ?, ?, ?, ?)',
            [userId, userRole, action, JSON.stringify(details), ipAddress]
        );
    } catch (err) {
        console.error('Audit log failed:', err.message);
    }
}

module.exports = { authenticate, requireAdmin, requireOperatorOrAdmin, auditLog };
