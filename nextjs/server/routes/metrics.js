const express = require('express');
const pool = require('../db/connection');

const router = express.Router();

// ── GET /api/metrics/kpi ─────────────────────────────────────────────────────
// PUBLIC endpoint — no authentication required. Returns all 6 platform KPIs
// computed from live inverter_readings data.
router.get('/kpi', async (req, res) => {
    try {
        const READING_INTERVAL_HRS = 5 / 3600; // readings every 5 seconds
        const EMISSION_FACTOR_KG = 0.71;      // kg CO₂ per kWh (global avg)
        const INVERTER_EFF = 0.96;      // nominal inverter AC/DC efficiency
        const TARIFF_USD = 0.06;      // USD per kWh (solar PPA tariff)
        const NOMINAL_TEMP = 45;        // °C nominal operating temp
        const CRITICAL_TEMP = 65;        // °C critical temp threshold
        const LOOKBACK_DAYS = 30;        // analysis window

        // ── 1. VCOY — Verified Carbon Offset Yield ──────────────────────────
        // Energy(kWh) = SUM(ac_power × 5/3600), CO₂ offset in tonnes
        const [[vcoyRow]] = await pool.query(`
            SELECT
                COALESCE(SUM(GREATEST(r.ac_power, 0) * ?), 0) AS energy_kwh,
                COALESCE(SUM(GREATEST(r.ac_power, 0) * ? * ?), 0) AS co2_kg
            FROM inverter_readings r
            WHERE r.timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)
        `, [READING_INTERVAL_HRS, READING_INTERVAL_HRS, EMISSION_FACTOR_KG, LOOKBACK_DAYS]);

        const energyKwh = Number(vcoyRow.energy_kwh) || 0;
        const co2Kg = Number(vcoyRow.co2_kg) || 0;
        const vcoy = co2Kg / 1000; // tonnes

        // Carbon credit revenue at $40/tonne
        const carbonRevenue = vcoy * 40;

        // ── 2. ELF — Energy Loss Due to Faults ──────────────────────────────
        // Expected_AC = dc_voltage * dc_current * 0.96
        // Loss occurs only when is_faulty = 1
        const [[elfRow]] = await pool.query(`
            SELECT
                COALESCE(SUM(
                    GREATEST((r.dc_voltage * r.dc_current * ?) - r.ac_power, 0) * ?
                ), 0) AS loss_kwh
            FROM inverter_readings r
            WHERE r.is_faulty = 1
              AND r.timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)
        `, [INVERTER_EFF, READING_INTERVAL_HRS, LOOKBACK_DAYS]);

        const elfKwh = Number(elfRow.loss_kwh) || 0;
        const elfRevenueLoss = elfKwh * TARIFF_USD;

        // ── 3. TDI — Thermal Degradation Index ──────────────────────────────
        // Stress = CLAMP((module_temp - 45) / (65 - 45), 0, 1)
        // TDI = AVG(Stress × ac_power / capacity_kw)
        const [[tdiRow]] = await pool.query(`
            SELECT
                COALESCE(AVG(
                    LEAST(GREATEST((r.module_temp - ?) / (? - ?), 0), 1)
                    * r.ac_power / GREATEST(i.capacity_kw, 1)
                ), 0) AS tdi
            FROM inverter_readings r
            JOIN inverters i ON i.id = r.inverter_id
            WHERE r.timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)
              AND r.module_temp > 0
        `, [NOMINAL_TEMP, CRITICAL_TEMP, NOMINAL_TEMP, LOOKBACK_DAYS]);

        const tdi = Math.min(Number(tdiRow.tdi) || 0, 1);

        // ── 4. SUE — Solar Utilization Efficiency ───────────────────────────
        // SUE = AVG(ac_power / (capacity_kw × irradiation / 1000))
        // Only valid rows where irradiation > 10 W/m² (exclude night)
        const [[sueRow]] = await pool.query(`
            SELECT
                COALESCE(AVG(
                    r.ac_power / GREATEST((i.capacity_kw * r.irradiation / 1000.0), 0.001)
                ), 0) AS sue
            FROM inverter_readings r
            JOIN inverters i ON i.id = r.inverter_id
            WHERE r.irradiation > 10
              AND r.ac_power >= 0
              AND r.timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)
        `, [LOOKBACK_DAYS]);

        const sue = Math.min(Number(sueRow.sue) || 0, 1);

        // ── 5. PPR — Plant Performance Ratio ────────────────────────────────
        // PPR = Total Actual Energy / Total Theoretical Energy
        // Theoretical = capacity_kw × irradiation / 1000 (per reading)
        const [[pprRow]] = await pool.query(`
            SELECT
                COALESCE(SUM(GREATEST(r.ac_power, 0) * ?), 0) AS actual_kwh,
                COALESCE(SUM(i.capacity_kw * r.irradiation / 1000.0 * ?), 0) AS theoretical_kwh
            FROM inverter_readings r
            JOIN inverters i ON i.id = r.inverter_id
            WHERE r.irradiation > 10
              AND r.timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)
        `, [READING_INTERVAL_HRS, READING_INTERVAL_HRS, LOOKBACK_DAYS]);

        const pprActual = Number(pprRow.actual_kwh) || 0;
        const pprTheoretical = Number(pprRow.theoretical_kwh) || 1;
        const ppr = Math.min(pprActual / pprTheoretical, 1);

        // ── 6. EIS — Environmental Impact Score ─────────────────────────────
        // EIS = 0.5 × Norm(CarbonOffset) + 0.3 × Efficiency − 0.2 × Downtime
        // Norm(CarbonOffset): clamp VCOY/100 to [0,1]
        // Efficiency: SUE
        // Downtime: fraction of faulty readings over total readings
        const [[downtimeRow]] = await pool.query(`
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(is_faulty = 1), 0) AS faulty
            FROM inverter_readings
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)
        `, [LOOKBACK_DAYS]);

        const totalReadings = Number(downtimeRow.total) || 1;
        const faultyReadings = Number(downtimeRow.faulty) || 0;
        const downtimeFrac = faultyReadings / totalReadings;

        const normCarbonOffset = Math.min(vcoy / 100, 1);
        const eis = Math.max(
            0.5 * normCarbonOffset + 0.3 * sue - 0.2 * downtimeFrac,
            0
        );

        // ── 7. Summary stats for display ────────────────────────────────────
        const [[inverterStats]] = await pool.query(`
            SELECT
                COUNT(*) AS total,
                SUM(current_category IN ('D','E','offline')) AS fault_or_offline
            FROM inverters
        `);

        const totalInverters = Number(inverterStats.total) || 0;
        const faultOrOffline = Number(inverterStats.fault_or_offline) || 0;
        const healthyPct = totalInverters > 0
            ? ((totalInverters - faultOrOffline) / totalInverters) * 100
            : 100;

        res.json({
            vcoy: {
                tonnes: Math.round(vcoy * 100) / 100,
                carbon_revenue_usd: Math.round(carbonRevenue * 100) / 100,
                energy_kwh_30d: Math.round(energyKwh),
            },
            elf: {
                loss_kwh: Math.round(elfKwh * 100) / 100,
                revenue_loss_usd: Math.round(elfRevenueLoss * 100) / 100,
            },
            tdi: {
                score: Math.round(tdi * 1000) / 1000,  // 0–1
                health_pct: Math.round((1 - tdi) * 100),
            },
            sue: {
                score: Math.round(sue * 1000) / 1000,  // 0–1
                percent: Math.round(sue * 100),
            },
            ppr: {
                score: Math.round(ppr * 1000) / 1000,  // 0–1
                percent: Math.round(ppr * 100),
            },
            eis: {
                score: Math.round(eis * 1000) / 1000,  // 0–1
                percent: Math.round(eis * 100),
            },
            fleet: {
                total_inverters: totalInverters,
                healthy_pct: Math.round(healthyPct),
                faulty_count: faultOrOffline,
            },
            computed_at: new Date().toISOString(),
        });
    } catch (err) {
        console.error('KPI metrics error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
