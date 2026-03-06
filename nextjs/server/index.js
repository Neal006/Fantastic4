require('dotenv').config();
const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const rateLimit = require('express-rate-limit');

const authRoutes = require('./routes/auth');
const operatorRoutes = require('./routes/operator');
const adminRoutes = require('./routes/admin');
const chatbotRoutes = require('./routes/chatbot');
const { startSimulator } = require('./simulator');

const app = express();
const PORT = process.env.PORT || 3001;

// ── CORS ──
app.use(cors({
    origin: ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
}));

// ── Body & Cookie Parsing ──
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// ── Rate Limiting (auth routes) ──
const authLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10,
    message: { error: 'Too many login attempts. Please wait a minute and try again.' },
    standardHeaders: true,
    legacyHeaders: false,
});

// ── Health Check ──
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString(), service: 'SolarWatch API' });
});

// ── Routes ──
app.use('/api/auth', authLimiter, authRoutes);
app.use('/api/operator', operatorRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/chatbot', chatbotRoutes);

// ── 404 Handler ──
app.use((req, res) => {
    res.status(404).json({ error: `Route ${req.method} ${req.path} not found` });
});

// ── Global Error Handler ──
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

// ── Start Server ──
app.listen(PORT, () => {
    console.log('');
    console.log('☀️  SolarWatch Backend');
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`📡 Health: http://localhost:${PORT}/health`);
    console.log('');

    // Start data simulator
    startSimulator();
});

module.exports = app;
