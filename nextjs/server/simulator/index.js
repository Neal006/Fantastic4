const pool = require('../db/connection');

// Categories weighted by probability
const CATEGORIES = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'E', 'offline'];

const FAULT_TYPES = {
    C: ['Low Efficiency', 'Minor Voltage Deviation', 'Partial Shading', 'High Module Temperature'],
    D: ['String Failure', 'MPPT Drift', 'Overcurrent Warning', 'String Voltage Mismatch'],
    E: ['Ground Fault', 'Arc Fault detected', 'Inverter Shutdown', 'DC Overcurrent', 'Insulation Failure'],
};

function rng(min, max) { return Math.random() * (max - min) + min; }
function rngFixed(min, max, dec = 2) { return parseFloat(rng(min, max).toFixed(dec)); }
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function assignCategory() {
    return pick(CATEGORIES);
}

function genShapValues(category) {
    if (category === 'A' || category === 'B') return null;
    const features = ['dc_voltage', 'dc_current', 'ac_power', 'module_temp', 'ambient_temp', 'irradiation'];
    const shap = {};
    for (const f of features) {
        shap[f] = parseFloat((category === 'E' ? rng(-0.3, 0.8) : rng(-0.2, 0.5)).toFixed(3));
    }
    return shap;
}

function genReading(category) {
    const isOffline = category === 'offline';
    const isFault = category === 'D' || category === 'E';

    const dcVoltage = isOffline ? 0 : isFault ? rngFixed(100, 350, 1) : rngFixed(550, 750, 1);
    const dcCurrent = isOffline ? 0 : isFault ? rngFixed(0.1, 6, 2) : rngFixed(5, 12, 2);
    const acPower = isOffline ? 0 : parseFloat((dcVoltage * dcCurrent * rng(0.85, 0.97) / 1000).toFixed(3));
    const moduleTemp = isOffline ? 25 : isFault ? rngFixed(65, 85, 1) : rngFixed(30, 55, 1);
    const ambientTemp = rngFixed(28, 42, 1);
    const irradiation = isOffline ? 0 : category === 'E' ? rngFixed(50, 400, 0) : rngFixed(600, 1050, 0);

    return { dcVoltage, dcCurrent, acPower, moduleTemp, ambientTemp, irradiation };
}

async function runSimulatorCycle() {
    try {
        // Get all inverters
        const [inverters] = await pool.query(`
      SELECT i.id, i.current_category, b.plant_id
      FROM inverters i
      JOIN blocks b ON i.block_id = b.id
    `);

        if (!inverters.length) return;

        for (const inverter of inverters) {
            const oldCategory = inverter.current_category;
            const newCategory = assignCategory();
            const reading = genReading(newCategory);
            const isFaulty = ['C', 'D', 'E'].includes(newCategory);
            const faultType = isFaulty ? pick(FAULT_TYPES[newCategory] || ['Unknown Fault']) : null;
            const confidence = newCategory === 'A' ? rngFixed(0.92, 0.99, 4)
                : newCategory === 'B' ? rngFixed(0.85, 0.94, 4)
                    : newCategory === 'C' ? rngFixed(0.60, 0.84, 4)
                        : rngFixed(0.30, 0.60, 4);

            const shapValues = genShapValues(newCategory);
            const isOnline = newCategory !== 'offline';

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

            // Update inverter current state
            await pool.query(
                `UPDATE inverters SET current_category = ?, is_online = ?, last_data_at = NOW() WHERE id = ?`,
                [newCategory, isOnline, inverter.id]
            );

            // Generate alert if category degraded to D or E (and wasn't already there)
            if ((newCategory === 'D' || newCategory === 'E') && oldCategory !== newCategory) {
                const alertType = newCategory === 'E' ? 'critical' : 'warning';
                await pool.query(
                    `INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to)
           VALUES (UUID(), ?, ?, ?, ?, ?)`,
                    [
                        inverter.id,
                        alertType,
                        `${faultType || 'Fault'} detected on inverter`,
                        oldCategory,
                        newCategory,
                    ]
                );
            }

            // Generate info alert for C category transition
            if (newCategory === 'C' && !['C', 'D', 'E'].includes(oldCategory)) {
                await pool.query(
                    `INSERT INTO alerts (id, inverter_id, type, message, category_from, category_to)
           VALUES (UUID(), ?, 'info', ?, ?, ?)`,
                    [inverter.id, `Performance warning: ${faultType || 'Minor deviation detected'}`, oldCategory, newCategory]
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
