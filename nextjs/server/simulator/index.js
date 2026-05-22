const pool = require('../db/connection');

// ── Service URLs (configurable for deployment on separate servers) ───────────
const GENAI_URL = process.env.GENAI_URL || 'http://localhost:8000';

// ── Realistic sensor ranges derived from ml/data/Plant 2 CSV files ──────────
// Source: Copy of 54-10-EC-8C-14-69.raws.csv & 54-10-EC-8C-14-6E.raws.csv
//
// CSV column            → API field       Observed range       Typical operating
// pv1_voltage           → dc_voltage      0 – 735 V           500 – 700 V
// pv1_current           → dc_current      0 – 102 A           30 – 80 A
// power                 → ac_power        0 – 65 kW           15 – 55 kW
// temp                  → module_temp     17 – 55 °C          35 – 50 °C
// ambient_temp          → ambient_temp    20 – 50 °C          28 – 42 °C
// meter_active_power    → irradiation     0 – 1200 W/m²       400 – 1000 W/m²
// freq                  → frequency       49.8 – 50.3 Hz
// pf                    → power_factor    0.85 – 0.99

// ── Scenario definitions per inverter (realistic failure modes) ─────────────
// Ranges based on CSV statistical profiles + physically plausible fault conditions
const SCENARIOS = {
    // ---- Plant 1 ----
    'INV-P1-L1-0': {
        type: 'healthy',
        dcV: [580, 680], dcA: [55, 80], acKw: [35, 55], modT: [35, 44], ambT: [28, 36],
        irr: [750, 1000], alarm: 0, opState: 5120, pf: [0.95, 0.99], freq: [49.95, 50.05],
    },
    'INV-P1-L1-1': {
        type: 'degradation',
        dcV: [420, 520], dcA: [15, 45], acKw: [12, 28], modT: [43, 48], ambT: [34, 38],
        irr: [600, 800], alarm: 0, opState: 5120, pf: [0.90, 0.96], freq: [49.96, 50.04],
    },
    'INV-P1-L2-0': {
        type: 'shutdown',
        dcV: [480, 560], dcA: [40, 55], acKw: [8, 18], modT: [62, 78], ambT: [42, 50],
        irr: [500, 750], alarm: 4003, opState: 0, pf: [0.80, 0.94], freq: [49.90, 50.08],
    },
    'INV-P1-L2-1': {
        type: 'healthy',
        dcV: [600, 700], dcA: [60, 82], acKw: [40, 58], modT: [36, 42], ambT: [29, 35],
        irr: [800, 1050], alarm: 0, opState: 5120, pf: [0.96, 0.99], freq: [49.97, 50.03],
    },
    // ---- Plant 2 ----
    'INV-P2-L1-0': {
        type: 'degradation',
        dcV: [500, 580], dcA: [30, 48], acKw: [18, 30], modT: [47, 54], ambT: [34, 40],
        irr: [550, 720], alarm: 3021, opState: 5120, pf: [0.88, 0.94], freq: [49.94, 50.04],
    },
    'INV-P2-L1-1': {
        type: 'healthy',
        dcV: [590, 690], dcA: [58, 78], acKw: [38, 52], modT: [38, 44], ambT: [30, 37],
        irr: [720, 950], alarm: 0, opState: 5120, pf: [0.94, 0.98], freq: [49.96, 50.04],
    },
    'INV-P2-L2-0': {
        type: 'degradation',
        dcV: [380, 560], dcA: [18, 55], acKw: [10, 25], modT: [40, 46], ambT: [33, 38],
        irr: [580, 780], alarm: 0, opState: 5120, pf: [0.85, 0.93], freq: [49.95, 50.03],
    },
    'INV-P2-L2-1': {
        type: 'shutdown',
        dcV: [0, 5], dcA: [0, 1], acKw: [0, 0.5], modT: [34, 40], ambT: [30, 36],
        irr: [100, 300], alarm: 5001, opState: 0, pf: [0.30, 0.50], freq: [50.8, 51.5],
    },
    // ---- Plant 3 ----
    'INV-P3-L1-0': {
        type: 'healthy',
        dcV: [620, 720], dcA: [65, 85], acKw: [42, 60], modT: [34, 41], ambT: [27, 33],
        irr: [850, 1100], alarm: 0, opState: 5120, pf: [0.97, 0.99], freq: [49.97, 50.03],
    },
    'INV-P3-L1-1': {
        type: 'healthy',
        dcV: [600, 700], dcA: [60, 80], acKw: [38, 54], modT: [37, 43], ambT: [28, 34],
        irr: [780, 1000], alarm: 0, opState: 5120, pf: [0.95, 0.99], freq: [49.96, 50.04],
    },
    'INV-P3-L2-0': {
        type: 'degradation',
        dcV: [420, 600], dcA: [22, 58], acKw: [15, 32], modT: [39, 44], ambT: [30, 36],
        irr: [620, 820], alarm: 0, opState: 5120, pf: [0.89, 0.95], freq: [49.96, 50.03],
    },
    'INV-P3-L2-1': {
        type: 'degradation',
        dcV: [520, 620], dcA: [45, 62], acKw: [25, 38], modT: [40, 46], ambT: [31, 36],
        irr: [650, 850], alarm: 2010, opState: 5120, pf: [0.88, 0.95], freq: [49.95, 50.04],
    },
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function rng(min, max) { return Math.random() * (max - min) + min; }
function rngFixed(min, max, dec = 2) { return parseFloat(rng(min, max).toFixed(dec)); }

/**
 * Generate a single sensor reading for an inverter based on its scenario.
 * Applies ±3% jitter on each cycle so live data feels dynamic.
 */
function generateReading(name) {
    const s = SCENARIOS[name];
    if (!s) return null;

    const jitter = v => v * (1 + (Math.random() - 0.5) * 0.06);

    return {
        inverter_id: name,
        dc_voltage: parseFloat(jitter(rngFixed(s.dcV[0], s.dcV[1])).toFixed(2)),
        dc_current: parseFloat(jitter(rngFixed(s.dcA[0], s.dcA[1])).toFixed(2)),
        ac_power:   parseFloat(jitter(rngFixed(s.acKw[0], s.acKw[1])).toFixed(2)),
        module_temp: parseFloat(jitter(rngFixed(s.modT[0], s.modT[1])).toFixed(2)),
        ambient_temp: parseFloat(jitter(rngFixed(s.ambT[0], s.ambT[1])).toFixed(2)),
        irradiation: parseFloat(jitter(rngFixed(s.irr[0], s.irr[1])).toFixed(2)),
        alarm_code: s.alarm,
        op_state: s.opState,
        power_factor: rngFixed(s.pf[0], s.pf[1], 3),
        frequency: rngFixed(s.freq[0], s.freq[1], 3),
    };
}

/**
 * Send readings to GenAI /simulate endpoint.
 * GenAI routes: 1 reading → ML /predict, >1 → ML /predict/batch.
 * Returns ML predictions with category, confidence, probabilities, SHAP, etc.
 */
async function callGenAISimulate(readings) {
    const url = `${GENAI_URL}/simulate`;
    const body = {
        readings,
        include_shap: true,
        include_plot: false,
    };

    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`GenAI /simulate returned ${resp.status}: ${text}`);
    }

    return resp.json();
}

// Map new ML text labels (no_risk/degradation/shutdown) to A-E grades for DB ENUM
function mapCategoryToGrade(raw, confidence) {
    if (['A', 'B', 'C', 'D', 'E', 'offline'].includes(raw)) return raw;
    if (raw === 'no_risk') return confidence >= 0.90 ? 'A' : 'B';
    if (raw === 'degradation' || raw === 'degradation_risk') return confidence >= 0.70 ? 'D' : 'C';
    if (raw === 'shutdown' || raw === 'shutdown_risk') return 'E';
    return 'A';
}

/**
 * Store an ML prediction result into the database and update inverter state.
 */
async function storePrediction(inverterId, dbInverterId, oldCategory, reading, prediction) {
    const confidence = prediction.confidence || 0.5;
    const category = mapCategoryToGrade(prediction.category || 'A', confidence);
    const probabilities = prediction.probabilities || {};
    const fault = prediction.fault || null;
    const isFaulty = ['C', 'D', 'E'].includes(category);
    const isOnline = category !== 'offline' && reading.op_state !== 0;

    // Build SHAP values array for DB storage
    let shapValues = null;
    if (prediction.shap && prediction.shap.top_features) {
        shapValues = JSON.stringify(prediction.shap.top_features);
    }

    try {
        await pool.query(
            `INSERT INTO inverter_readings
            (id, inverter_id, timestamp, category, confidence, probabilities, is_faulty, fault_type,
             dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values)
           VALUES (UUID(), ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
                dbInverterId, category, confidence,
                JSON.stringify(probabilities),
                isFaulty, fault,
                reading.dc_voltage, reading.dc_current, reading.ac_power,
                reading.module_temp, reading.ambient_temp, reading.irradiation,
                shapValues,
            ]
        );
    } catch (colErr) {
        if (colErr.code === 'ER_BAD_FIELD_ERROR') {
            await pool.query(
                `INSERT INTO inverter_readings
                (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
                 dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values)
               VALUES (UUID(), ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    dbInverterId, category, confidence, isFaulty, fault,
                    reading.dc_voltage, reading.dc_current, reading.ac_power,
                    reading.module_temp, reading.ambient_temp, reading.irradiation,
                    shapValues,
                ]
            );
        } else {
            throw colErr;
        }
    }

    await pool.query(
        `UPDATE inverters SET current_category = ?, is_online = ?, last_data_at = NOW() WHERE id = ?`,
        [category, isOnline ? 1 : 0, dbInverterId]
    );

    // Generate alerts on category transitions
    if ((category === 'D' || category === 'E') && oldCategory !== category) {
        const alertType = category === 'E' ? 'critical' : 'warning';
        await pool.query(
            `INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to)
           VALUES (UUID(), ?, ?, ?, ?, ?)`,
            [dbInverterId, alertType, `${fault || 'Fault'} detected on ${inverterId}`, oldCategory, category]
        );
    }
    if (category === 'C' && !['C', 'D', 'E'].includes(oldCategory)) {
        await pool.query(
            `INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to)
           VALUES (UUID(), ?, 'info', ?, ?, ?)`,
            [dbInverterId, `Performance warning: ${fault || 'Minor deviation'} on ${inverterId}`, oldCategory, category]
        );
    }
}

// ── Main simulation cycle ───────────────────────────────────────────────────
let mlPipelineAvailable = true;

async function runSimulatorCycle() {
    try {
        const [inverters] = await pool.query(`
            SELECT i.id, i.name, i.current_category, b.plant_id
            FROM inverters i
            JOIN blocks b ON i.block_id = b.id
        `);

        if (!inverters.length) return;

        // Generate sensor readings for all inverters
        const allReadings = [];
        const inverterMap = {}; // name → { dbId, oldCategory }
        for (const inv of inverters) {
            const reading = generateReading(inv.name);
            if (reading) {
                allReadings.push(reading);
                inverterMap[inv.name] = { dbId: inv.id, oldCategory: inv.current_category };
            }
        }

        if (!allReadings.length) return;

        // ── Try ML pipeline: Simulator → GenAI → ML Inference ──
        if (mlPipelineAvailable) {
            try {
                const result = await callGenAISimulate(allReadings);
                const predictions = result.predictions || [];

                // Build a lookup from inverter_id → prediction
                const predMap = {};
                for (const pred of predictions) {
                    predMap[pred.inverter_id] = pred;
                }

                // Store each prediction
                for (const reading of allReadings) {
                    const inv = inverterMap[reading.inverter_id];
                    if (!inv) continue;
                    const prediction = predMap[reading.inverter_id];
                    if (prediction) {
                        await storePrediction(reading.inverter_id, inv.dbId, inv.oldCategory, reading, prediction);
                    }
                }

                console.log(`[Simulator] ✅ ML pipeline: ${allReadings.length} inverters → GenAI → ML at ${new Date().toISOString()}`);
                return; // Success — skip fallback
            } catch (mlErr) {
                if (mlPipelineAvailable) {
                    console.warn(`[Simulator] ⚠️ ML pipeline unavailable (${mlErr.message}). Using local fallback.`);
                    mlPipelineAvailable = false;
                    // Retry after 60s
                    setTimeout(() => { mlPipelineAvailable = true; }, 60000);
                }
            }
        }

        // ── Fallback: local simulation (no ML server) ──
        for (const reading of allReadings) {
            const inv = inverterMap[reading.inverter_id];
            if (!inv) continue;
            const scenario = SCENARIOS[reading.inverter_id];
            const type = scenario?.type || 'healthy';

            // Generate a local prediction based on scenario type
            let category, confidence, probabilities, fault;
            if (type === 'healthy') {
                category = 'A';
                confidence = rngFixed(0.90, 0.99, 4);
                probabilities = { no_risk: confidence, degradation_risk: rngFixed(0.005, 0.05, 3), shutdown_risk: rngFixed(0.001, 0.02, 3) };
                fault = null;
            } else if (type === 'degradation') {
                category = Math.random() < 0.6 ? 'D' : 'C';
                confidence = rngFixed(0.50, 0.80, 4);
                const dr = rngFixed(0.40, 0.65, 3);
                const nr = rngFixed(0.15, 0.35, 3);
                probabilities = { no_risk: nr, degradation_risk: dr, shutdown_risk: parseFloat(Math.max(0, 1 - nr - dr).toFixed(3)) };
                fault = scenario.alarm ? `Alarm Code ${scenario.alarm} — Operational Fault` : 'String Degradation';
            } else {
                category = 'E';
                confidence = rngFixed(0.35, 0.60, 4);
                const sr = rngFixed(0.55, 0.80, 3);
                const dr = rngFixed(0.10, 0.25, 3);
                probabilities = { no_risk: parseFloat(Math.max(0, 1 - sr - dr).toFixed(3)), degradation_risk: dr, shutdown_risk: sr };
                fault = reading.module_temp > 60 ? 'Overheating — Thermal Shutdown Risk' : 'Grid Fault — Frequency Deviation';
            }

            const localPrediction = { category, confidence, probabilities, fault, shap: null };
            await storePrediction(reading.inverter_id, inv.dbId, inv.oldCategory, reading, localPrediction);
        }

        console.log(`[Simulator] ✅ Local fallback: ${allReadings.length} inverters at ${new Date().toISOString()}`);
    } catch (err) {
        console.error('[Simulator] ❌ Error:', err.message);
    }
}

function startSimulator() {
    const interval = parseInt(process.env.SIMULATOR_INTERVAL_MS) || 15000;
    console.log(`[Simulator] 🚀 Starting — updates every ${interval / 1000}s`);
    console.log(`[Simulator] 📡 GenAI URL: ${GENAI_URL}`);
    console.log(`[Simulator] 🔄 Pipeline: Simulator → GenAI /simulate → ML Inference → DB`);
    setTimeout(runSimulatorCycle, 3000);
    setInterval(runSimulatorCycle, interval);
}

module.exports = { startSimulator };
