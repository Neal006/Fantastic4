const express = require('express');
const { authenticate } = require('../middleware/auth');
const pool = require('../db/connection');

const router = express.Router();
router.use(authenticate);

const GENAI_URL = process.env.GENAI_URL || 'http://localhost:8000';

/**
 * Generic helper: forward a request to the GenAI FastAPI server and pipe the
 * response back to Express. Returns the parsed JSON body on success or throws
 * an Error with a readable message on failure.
 */
async function callGenAI(method, path, body) {
    const url = `${GENAI_URL}${path}`;
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    const text = await res.text();
    if (!res.ok) {
        let detail = text;
        try { detail = JSON.parse(text)?.detail || text; } catch (_) { /* ignored */ }
        throw Object.assign(new Error(detail), { status: res.status });
    }
    try { return JSON.parse(text); } catch (_) { return text; }
}

/**
 * Fire-and-forget helper: persist a chat turn to the chat_logs table.
 * Never throws — a DB failure must not affect the API response.
 *
 * @param {string}      sessionId   - groups messages into one conversation
 * @param {object}      user        - req.user (id, role)
 * @param {'user'|'assistant'} role - which side of the conversation
 * @param {string}      message     - the text content
 * @param {object}      [opts]      - optional { inverter_id, metadata }
 */
async function saveChatLog(sessionId, user, role, message, opts = {}) {
    try {
        const userId = user?.id || null;
        const userRole = user?.role || 'public'; // 'admin' | 'operator' | 'public'
        await pool.query(
            `INSERT INTO chat_logs
               (id, session_id, user_id, user_role, role, message, inverter_id, metadata)
             VALUES (UUID(), ?, ?, ?, ?, ?, ?, ?)`,
            [
                sessionId,
                userId,
                userRole,
                role,
                message,
                opts.inverter_id || null,
                opts.metadata ? JSON.stringify(opts.metadata) : null,
            ]
        );
    } catch (err) {
        // Log but never surface to the caller
        console.error('[ChatLog] Failed to save chat log:', err.message);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/health
// ─────────────────────────────────────────────────────────────────────────────
router.get('/health', async (req, res) => {
    try {
        const data = await callGenAI('GET', '/health');
        res.json(data);
    } catch (err) {
        console.error('[Chatbot] Health check failed:', err.message);
        res.status(503).json({ error: 'GenAI service unavailable', detail: err.message });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  POST /api/chatbot/query
//  Multi-turn RAG conversational Q&A. Stores user message + AI response.
//  Body: { message, session_id? }
// ─────────────────────────────────────────────────────────────────────────────
router.post('/query', async (req, res) => {
    try {
        const { message, session_id } = req.body;
        if (!message) return res.status(400).json({ error: 'Message is required' });

        // Use the provided session_id or generate one for a new conversation
        const sessionId = session_id || require('crypto').randomUUID();

        // Persist the USER message before calling GenAI
        saveChatLog(sessionId, req.user, 'user', message);

        const data = await callGenAI('POST', '/chat', { message, session_id: sessionId });

        const response = {
            response: data.response,
            session_id: data.session_id || sessionId,
            sources_used: data.sources_used || [],
        };

        // Persist the ASSISTANT response
        saveChatLog(response.session_id, req.user, 'assistant', data.response, {
            metadata: { sources_used: data.sources_used || [] },
        });

        res.json(response);
    } catch (err) {
        console.error('[Chatbot] Query error:', err.message);
        res.status(err.status || 500).json({ error: 'The AI assistant is currently unavailable. Please try again.' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/explanation/:inverterId
//  AI plain-English risk explanation. Logs the explanation as an event.
// ─────────────────────────────────────────────────────────────────────────────
router.get('/explanation/:inverterId', async (req, res) => {
    try {
        const { inverterId } = req.params;
        const data = await callGenAI('GET', `/explanation/${encodeURIComponent(inverterId)}`);

        // Log the explanation fetch as a system-generated assistant message
        const sessionId = require('crypto').randomUUID();
        saveChatLog(sessionId, req.user, 'assistant', JSON.stringify(data), {
            inverter_id: null, // we only have the name, not the UUID
            metadata: { action: 'explanation', inverter_name: inverterId },
        });

        res.json(data);
    } catch (err) {
        console.error('[Chatbot] Explanation error:', err.message);
        const status = err.status === 404 ? 404 : 500;
        res.status(status).json({ error: err.message || 'Failed to get AI explanation' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  POST /api/chatbot/ticket/:inverterId
//  Generate a maintenance ticket. Logs the generation event.
// ─────────────────────────────────────────────────────────────────────────────
router.post('/ticket/:inverterId', async (req, res) => {
    try {
        const { inverterId } = req.params;
        const data = await callGenAI('POST', `/agent/maintenance-ticket/${encodeURIComponent(inverterId)}`);

        const sessionId = require('crypto').randomUUID();
        saveChatLog(sessionId, req.user, 'user', `[Maintenance ticket requested for ${inverterId}]`, {
            metadata: { action: 'ticket_request', inverter_name: inverterId },
        });
        saveChatLog(sessionId, req.user, 'assistant', JSON.stringify(data), {
            metadata: { action: 'ticket_generated', inverter_name: inverterId },
        });

        res.json(data);
    } catch (err) {
        console.error('[Chatbot] Ticket generation error:', err.message);
        const status = err.status === 404 ? 404 : 500;
        res.status(status).json({ error: err.message || 'Failed to generate maintenance ticket' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/ticket/:inverterId/pdf
//  Download the maintenance ticket PDF — streams bytes from GenAI server
// ─────────────────────────────────────────────────────────────────────────────
router.get('/ticket/:inverterId/pdf', async (req, res) => {
    try {
        const { inverterId } = req.params;
        const url = `${GENAI_URL}/agent/maintenance-ticket/${encodeURIComponent(inverterId)}/pdf`;
        const upstream = await fetch(url, { method: 'GET' });
        if (!upstream.ok) {
            const text = await upstream.text();
            let detail = text;
            try { detail = JSON.parse(text)?.detail || text; } catch (_) { /* ignored */ }
            return res.status(upstream.status).json({ error: detail });
        }
        // Log the PDF download event
        saveChatLog(require('crypto').randomUUID(), req.user, 'user',
            `[PDF downloaded for ${inverterId}]`,
            { metadata: { action: 'pdf_download', inverter_name: inverterId } }
        );
        const contentDisposition = upstream.headers.get('content-disposition') || `attachment; filename="${inverterId}.pdf"`;
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', contentDisposition);
        const buffer = await upstream.arrayBuffer();
        res.send(Buffer.from(buffer));
    } catch (err) {
        console.error('[Chatbot] PDF download error:', err.message);
        res.status(500).json({ error: 'Failed to download maintenance ticket PDF' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/predictions/:inverterId
//  Raw ML prediction for a single inverter
// ─────────────────────────────────────────────────────────────────────────────
router.get('/predictions/:inverterId', async (req, res) => {
    try {
        const { inverterId } = req.params;
        const data = await callGenAI('GET', `/predictions/${encodeURIComponent(inverterId)}`);
        res.json(data);
    } catch (err) {
        const status = err.status === 404 ? 404 : 500;
        res.status(status).json({ error: err.message || 'Failed to get prediction' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/logs
//  Retrieve chat logs — admins see all, operators see their own
//  Query params: ?limit=50&offset=0&user_id=<uuid>
// ─────────────────────────────────────────────────────────────────────────────
router.get('/logs', async (req, res) => {
    try {
        const limit = Math.min(parseInt(req.query.limit) || 50, 200);
        const offset = parseInt(req.query.offset) || 0;

        let query, params;

        if (req.user.role === 'admin') {
            // Admins can filter by any user_id or see everything
            const filterUserId = req.query.user_id || null;
            if (filterUserId) {
                query = `SELECT * FROM chat_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?`;
                params = [filterUserId, limit, offset];
            } else {
                query = `SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT ? OFFSET ?`;
                params = [limit, offset];
            }
        } else {
            // Operators only see their own logs
            query = `SELECT * FROM chat_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?`;
            params = [req.user.id, limit, offset];
        }

        const [logs] = await pool.query(query, params);
        res.json(logs);
    } catch (err) {
        console.error('[Chatbot] Logs error:', err.message);
        res.status(500).json({ error: 'Failed to retrieve chat logs' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  POST /api/chatbot/predict-test
//  Manual single-datapoint ML prediction for testing / demo.
//  Body: { inverter_id, dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, ... }
// ─────────────────────────────────────────────────────────────────────────────
router.post('/predict-test', async (req, res) => {
    const ML_URL = process.env.ML_INFERENCE_URL || 'http://localhost:8001';
    try {
        const { inverter_id, dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation,
                alarm_code, op_state, power_factor, frequency } = req.body;

        if (dc_voltage == null || dc_current == null || ac_power == null ||
            module_temp == null || ambient_temp == null || irradiation == null) {
            return res.status(400).json({ error: 'All 6 core fields are required: dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation' });
        }

        // Call ML inference directly (skip GenAI for speed)
        const mlRes = await fetch(`${ML_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                inverter_id: inverter_id || 'TEST-INV',
                dc_voltage: Number(dc_voltage),
                dc_current: Number(dc_current),
                ac_power: Number(ac_power),
                module_temp: Number(module_temp),
                ambient_temp: Number(ambient_temp),
                irradiation: Number(irradiation),
                alarm_code: Number(alarm_code) || 0,
                op_state: Number(op_state) || 5120,
                power_factor: power_factor != null ? Number(power_factor) : undefined,
                frequency: frequency != null ? Number(frequency) : undefined,
                include_shap: true,
                include_plot: false,
            }),
        });
        const mlText = await mlRes.text();
        if (!mlRes.ok) {
            let detail = mlText;
            try { detail = JSON.parse(mlText)?.detail || mlText; } catch (_) {}
            throw Object.assign(new Error(detail), { status: mlRes.status });
        }
        const prediction = JSON.parse(mlText);
        res.json({ count: 1, predictions: [prediction], timestamp: new Date().toISOString() });
    } catch (err) {
        console.error('[Chatbot] Predict-test error:', err.message);
        res.status(err.status || 500).json({ error: err.message || 'ML prediction failed' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/reference-pdf
//  Download the reference analysis report PDF
// ─────────────────────────────────────────────────────────────────────────────
router.get('/reference-pdf', (req, res) => {
    const path = require('path');
    const pdfPath = path.join(__dirname, '..', 'reference-report.pdf');
    const fs = require('fs');
    if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ error: 'Reference PDF not found' });
    }
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename="SolarWatch-Analysis-Report.pdf"');
    fs.createReadStream(pdfPath).pipe(res);
});

module.exports = router;
