const express = require('express');
const { authenticate } = require('../middleware/auth');

const router = express.Router();
router.use(authenticate);

const GENAI_URL = process.env.GENAI_URL || 'http://localhost:8000';

/**
 * Generic helper: forward a request to the GenAI FastAPI server and pipe the
 * response back to Express.  Returns the parsed JSON body on success or throws
 * an Error with a readable message on failure.
 */
async function callGenAI(method, path, body) {
    const url = `${GENAI_URL}${path}`;
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) {
        opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const text = await res.text();
    if (!res.ok) {
        let detail = text;
        try { detail = JSON.parse(text)?.detail || text; } catch (_) { /* ignored */ }
        throw Object.assign(new Error(detail), { status: res.status });
    }
    try { return JSON.parse(text); } catch (_) { return text; }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/health
//  Proxy the GenAI server health check
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
//  Multi-turn RAG-augmented conversational Q&A (proxies to POST /chat)
//  Body: { message, session_id? }
// ─────────────────────────────────────────────────────────────────────────────
router.post('/query', async (req, res) => {
    try {
        const { message, session_id } = req.body;
        if (!message) return res.status(400).json({ error: 'Message is required' });

        const data = await callGenAI('POST', '/chat', { message, session_id: session_id || null });
        // Return: { session_id, response, sources_used }
        res.json({
            response: data.response,
            session_id: data.session_id,
            sources_used: data.sources_used || [],
        });
    } catch (err) {
        console.error('[Chatbot] Query error:', err.message);
        res.status(err.status || 500).json({ error: 'The AI assistant is currently unavailable. Please try again.' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/explanation/:inverterId
//  AI plain-English risk explanation for a single inverter
// ─────────────────────────────────────────────────────────────────────────────
router.get('/explanation/:inverterId', async (req, res) => {
    try {
        const { inverterId } = req.params;
        const data = await callGenAI('GET', `/explanation/${encodeURIComponent(inverterId)}`);
        res.json(data);
    } catch (err) {
        console.error('[Chatbot] Explanation error:', err.message);
        const status = err.status === 404 ? 404 : 500;
        res.status(status).json({ error: err.message || 'Failed to get AI explanation' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  POST /api/chatbot/ticket/:inverterId
//  Generate a maintenance ticket (JSON) for an inverter
// ─────────────────────────────────────────────────────────────────────────────
router.post('/ticket/:inverterId', async (req, res) => {
    try {
        const { inverterId } = req.params;
        const data = await callGenAI('POST', `/agent/maintenance-ticket/${encodeURIComponent(inverterId)}`);
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
        // Stream the PDF bytes straight through to the client
        const contentDisposition = upstream.headers.get('content-disposition') || `attachment; filename="${inverterId}.pdf"`;
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', contentDisposition);
        // Node 18+ fetch returns a web ReadableStream; pipe via arrayBuffer for simplicity
        const buffer = await upstream.arrayBuffer();
        res.send(Buffer.from(buffer));
    } catch (err) {
        console.error('[Chatbot] PDF download error:', err.message);
        res.status(500).json({ error: 'Failed to download maintenance ticket PDF' });
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  GET /api/chatbot/predictions/:inverterId
//  Raw ML prediction for a single inverter (useful for the frontend)
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

module.exports = router;
