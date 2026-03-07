const pool = require('../db/connection');

// ── Scenario-aware sensor data per inverter name ─────────────────────────────
// Keeps readings physically plausible for each real failure scenario.
// Values derived from genai/app/synthetic_data.py raw_features.
const SCENARIOS = {
    'INV-P1-L1-0': { cat: 'A', dcV: [37.5, 39.0], dcA: [9.5, 10.0], acKw: [8.5, 9.0], modT: [40, 44], ambT: [32, 36], irr: [890, 960], fault: null },
    'INV-P1-L1-1': { cat: 'D', dcV: [27.0, 29.0], dcA: [3.0, 9.6], acKw: [5.5, 6.8], modT: [43, 46], ambT: [34, 37], irr: [780, 840], fault: 'String Degradation' },
    'INV-P1-L2-0': { cat: 'E', dcV: [35.5, 37.0], dcA: [7.8, 8.2], acKw: [3.5, 4.5], modT: [75, 82], ambT: [45, 49], irr: [700, 780], fault: 'Overheating — Thermal Shutdown Risk' },
    'INV-P1-L2-1': { cat: 'A', dcV: [37.8, 38.8], dcA: [9.7, 10.0], acKw: [8.8, 9.2], modT: [39, 42], ambT: [32, 35], irr: [920, 970], fault: null },
    'INV-P2-L1-0': { cat: 'D', dcV: [34.5, 35.8], dcA: [6.5, 7.0], acKw: [5.0, 5.8], modT: [49, 53], ambT: [35, 38], irr: [680, 730], fault: 'Alarm Code 3021 — Operational Fault' },
    'INV-P2-L1-1': { cat: 'B', dcV: [37.5, 38.2], dcA: [9.4, 9.7], acKw: [8.6, 9.0], modT: [40, 43], ambT: [34, 36], irr: [890, 930], fault: null },
    'INV-P2-L2-0': { cat: 'C', dcV: [29.5, 37.8], dcA: [4.0, 9.4], acKw: [4.8, 5.4], modT: [42, 45], ambT: [35, 37], irr: [760, 810], fault: 'Low Power Output — String Issue' },
    'INV-P2-L2-1': { cat: 'E', dcV: [0, 0], dcA: [0, 0], acKw: [0, 0], modT: [37, 40], ambT: [33, 36], irr: [150, 250], fault: 'Grid Fault — Frequency Deviation' },
    'INV-P3-L1-0': { cat: 'A', dcV: [38.0, 39.0], dcA: [9.8, 10.0], acKw: [9.0, 9.4], modT: [38, 41], ambT: [30, 33], irr: [940, 980], fault: null },
    'INV-P3-L1-1': { cat: 'A', dcV: [37.5, 38.5], dcA: [9.6, 9.8], acKw: [8.6, 9.0], modT: [40, 42], ambT: [31, 34], irr: [910, 950], fault: null },
    'INV-P3-L2-0': { cat: 'C', dcV: [30.5, 38.0], dcA: [4.5, 9.5], acKw: [6.8, 7.4], modT: [41, 43], ambT: [32, 35], irr: [780, 820], fault: 'Partial Shading — Strings 7 & 8' },
    'INV-P3-L2-1': { cat: 'C', dcV: [36.5, 37.5], dcA: [8.5, 8.9], acKw: [6.5, 7.0], modT: [42, 44], ambT: [33, 35], irr: [810, 850], fault: 'Communication Issue — Alarm Code 2010' },
};

// Category confidence ranges per category level
const CONFIDENCE_RANGE = {
    A: [0.92, 0.99], B: [0.85, 0.94], C: [0.55, 0.84], D: [0.40, 0.65], E: [0.30, 0.55], offline: [0.10, 0.30],
};

// Fault types for random simulation when category drifts
const FAULT_TYPES = {
    C: ['Low Efficiency', 'Minor Voltage Deviation', 'Partial Shading', 'High Module Temperature'],
    D: ['String Failure', 'MPPT Drift', 'Overcurrent Warning', 'String Voltage Mismatch'],
    E: ['Ground Fault', 'Arc Fault detected', 'Inverter Shutdown', 'DC Overcurrent'],
};

function rng(min, max) { return Math.random() * (max - min) + min; }
function rngFixed(min, max, dec = 2) { return parseFloat(rng(min, max).toFixed(dec)); }
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function genShapValues(category) {
    if (category === 'A' || category === 'B' || category === 'offline') return null;
    // Match the ML model output format: { feature, value }
    // Feature names mirror the model's engineered features
    const features = [
        'power_rmean_24h',
        'temp_rstd_6h',
        'dc_voltage',
        'dc_current',
        'ac_power',
        'module_temp',
        'ambient_temp',
        'irradiation',
    ];
    const shap = features.map(f => ({
        feature: f,
        value: parseFloat((category === 'E' ? rng(-0.3, 0.8) : rng(-0.2, 0.5)).toFixed(3)),
    }));
    // Sort descending by absolute value so top contributor is first
    shap.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return shap;
}

/**
 * Generate ML-model-style probability distribution matching the model JSON output:
 * { no_risk, degradation_risk, shutdown_risk }
 * Values sum to 1.0.
 */
function genProbabilities(category, confidence) {
    const c = parseFloat(confidence);
    if (category === 'A') {
        return { no_risk: c, degradation_risk: parseFloat(rng(0.005, 1 - c - 0.005).toFixed(3)), shutdown_risk: parseFloat((1 - c - rng(0.005, 0.02)).toFixed(3)) };
    }
    if (category === 'B') {
        const nr = parseFloat(rng(0.70, 0.89).toFixed(3));
        const dr = parseFloat(rng(0.08, 0.25).toFixed(3));
        return { no_risk: nr, degradation_risk: dr, shutdown_risk: parseFloat((1 - nr - dr).toFixed(3)) };
    }
    if (category === 'C') {
        const nr = parseFloat(rng(0.30, 0.55).toFixed(3));
        const dr = parseFloat(rng(0.35, 0.55).toFixed(3));
        return { no_risk: nr, degradation_risk: dr, shutdown_risk: parseFloat(Math.max(0, 1 - nr - dr).toFixed(3)) };
    }
    if (category === 'D') {
        const nr = parseFloat(rng(0.05, 0.20).toFixed(3));
        const dr = parseFloat(rng(0.50, 0.70).toFixed(3));
        return { no_risk: nr, degradation_risk: dr, shutdown_risk: parseFloat(Math.max(0, 1 - nr - dr).toFixed(3)) };
    }
    // E or offline — high shutdown risk
    const nr = parseFloat(rng(0.01, 0.08).toFixed(3));
    const dr = parseFloat(rng(0.10, 0.25).toFixed(3));
    return { no_risk: nr, degradation_risk: dr, shutdown_risk: parseFloat(Math.max(0, 1 - nr - dr).toFixed(3)) };
}

function genReadingForScenario(scenario, jitter = true) {
    // Apply ±5% jitter to keep live data feeling dynamic
    const j = v => jitter ? v * (1 + (Math.random() - 0.5) * 0.05) : v;

    const dcVoltage = parseFloat(j(rngFixed(scenario.dcV[0], scenario.dcV[1])).toFixed(2));
    const dcCurrent = parseFloat(j(rngFixed(scenario.dcA[0], scenario.dcA[1])).toFixed(4));
    const acPower = parseFloat(j(rngFixed(scenario.acKw[0], scenario.acKw[1])).toFixed(3));
    const moduleTemp = parseFloat(j(rngFixed(scenario.modT[0], scenario.modT[1])).toFixed(2));
    const ambientTemp = parseFloat(j(rngFixed(scenario.ambT[0], scenario.ambT[1])).toFixed(2));
    const irradiation = parseFloat(j(rngFixed(scenario.irr[0], scenario.irr[1])).toFixed(2));
    return { dcVoltage, dcCurrent, acPower, moduleTemp, ambientTemp, irradiation };
}

async function runSimulatorCycle() {
    try {
        // Fetch all inverters with their name so we can match scenarios
        const [inverters] = await pool.query(`
      SELECT i.id, i.name, i.current_category, b.plant_id
      FROM inverters i
      JOIN blocks b ON i.block_id = b.id
    `);

        if (!inverters.length) return;

        for (const inverter of inverters) {
            const scenario = SCENARIOS[inverter.name];
            const oldCategory = inverter.current_category;

            // Use scenario-locked category (with rare 5% drift chance for realism)
            const locked = scenario?.cat ?? 'A';
            const newCategory = (Math.random() < 0.05 && locked !== 'E' && locked !== 'offline')
                ? (locked === 'A' ? 'B' : locked === 'D' ? 'C' : locked)
                : locked;

            const isOnline = newCategory !== 'offline';
            const isFaulty = ['C', 'D', 'E'].includes(newCategory);
            const faultType = scenario?.fault ?? (isFaulty ? pick(FAULT_TYPES[newCategory] || ['Unknown Fault']) : null);

            const confRange = CONFIDENCE_RANGE[newCategory] || [0.5, 0.8];
            const confidence = rngFixed(confRange[0], confRange[1], 4);

            const reading = scenario
                ? genReadingForScenario(scenario)
                : { dcVoltage: 0, dcCurrent: 0, acPower: 0, moduleTemp: 25, ambientTemp: 30, irradiation: 0 };

            // For known inverters use real SHAP labels; for others generate generic ones
            const shapValues = isFaulty ? genShapValues(newCategory) : null;

            // Generate ML-model-style probability distribution
            const probabilities = genProbabilities(newCategory, confidence);

            // Use try/catch so older schema (without probabilities column) still works
            try {
                await pool.query(
                    `INSERT INTO inverter_readings
            (id, inverter_id, timestamp, category, confidence, probabilities, is_faulty, fault_type,
             dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values)
           VALUES (UUID(), ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                    [
                        inverter.id, newCategory, confidence,
                        JSON.stringify(probabilities),
                        isFaulty, faultType,
                        reading.dcVoltage, reading.dcCurrent, reading.acPower,
                        reading.moduleTemp, reading.ambientTemp, reading.irradiation,
                        shapValues ? JSON.stringify(shapValues) : null,
                    ]
                );
            } catch (colErr) {
                // Fallback: old schema without probabilities column
                if (colErr.code === 'ER_BAD_FIELD_ERROR') {
                    await pool.query(
                        `INSERT INTO inverter_readings
              (id, inverter_id, timestamp, category, confidence, is_faulty, fault_type,
               dc_voltage, dc_current, ac_power, module_temp, ambient_temp, irradiation, shap_values)
             VALUES (UUID(), ?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                        [
                            inverter.id, newCategory, confidence, isFaulty, faultType,
                            reading.dcVoltage, reading.dcCurrent, reading.acPower,
                            reading.moduleTemp, reading.ambientTemp, reading.irradiation,
                            shapValues ? JSON.stringify(shapValues) : null,
                        ]
                    );
                } else {
                    throw colErr;
                }
            }

            await pool.query(
                `UPDATE inverters SET current_category = ?, is_online = ?, last_data_at = NOW() WHERE id = ?`,
                [newCategory, isOnline, inverter.id]
            );

            // Generate alert on category degradation to D or E
            if ((newCategory === 'D' || newCategory === 'E') && oldCategory !== newCategory) {
                const alertType = newCategory === 'E' ? 'critical' : 'warning';
                await pool.query(
                    `INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to)
           VALUES (UUID(), ?, ?, ?, ?, ?)`,
                    [inverter.id, alertType, `${faultType || 'Fault'} detected on ${inverter.name}`, oldCategory, newCategory]
                );
            }

            // Info alert on C transition
            if (newCategory === 'C' && !['C', 'D', 'E'].includes(oldCategory)) {
                await pool.query(
                    `INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to)
           VALUES (UUID(), ?, 'info', ?, ?, ?)`,
                    [inverter.id, `Performance warning: ${faultType || 'Minor deviation'} on ${inverter.name}`, oldCategory, newCategory]
                );
            }
        }

        console.log(`[Simulator] ✅ Updated ${inverters.length} inverters at ${new Date().toISOString()}`);
    } catch (err) {
        console.error('[Simulator] ❌ Error:', err.message);
    }
}

function startSimulator() {
    const interval = parseInt(process.env.SIMULATOR_INTERVAL_MS) || 15000;
    console.log(`[Simulator] 🚀 Starting — updates every ${interval / 1000}s`);
    setTimeout(runSimulatorCycle, 2000);
    setInterval(runSimulatorCycle, interval);
}

module.exports = { startSimulator };
