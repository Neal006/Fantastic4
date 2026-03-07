/**
 * Seed script for Hackamined_new database
 * Run with: node db/seed_new.js
 *
 * Creates:
 *  - 1 Admin  : admin@lumin.ai      / Admin@123!
 *  - 1 Operator: arjun.mehta@lumin.ai / Op@12345
 *  - 3 Plants (Plant 1, 2, 3)
 *  - 2 Blocks per plant  (Block A, Block B)
 *  - 2 Inverters per block (matching SCENARIOS in simulator)
 *  - Default settings row
 */

require('dotenv').config();
const bcrypt = require('bcryptjs');
const mysql = require('mysql2/promise');

const SALT_ROUNDS = 12;

// ── Credentials ─────────────────────────────────────────────────────────────
const ADMIN = {
    name: 'System Admin',
    email: 'admin@lumin.ai',
    pass: 'Admin@123!',
};

const OPERATOR = {
    name: 'Arjun Mehta',
    email: 'arjun.mehta@lumin.ai',
    pass: 'Op@12345',
};

// ── Plant / Block / Inverter layout ─────────────────────────────────────────
// Must match the SCENARIOS object in simulator/index.js
const PLANTS = [
    {
        name: 'Plant 1', location: 'Rajasthan, India', capacity: 200,
        blocks: [
            { name: 'Block A (L1)', inverters: ['INV-P1-L1-0', 'INV-P1-L1-1'] },
            { name: 'Block B (L2)', inverters: ['INV-P1-L2-0', 'INV-P1-L2-1'] },
        ],
    },
    {
        name: 'Plant 2', location: 'Gujarat, India', capacity: 200,
        blocks: [
            { name: 'Block A (L1)', inverters: ['INV-P2-L1-0', 'INV-P2-L1-1'] },
            { name: 'Block B (L2)', inverters: ['INV-P2-L2-0', 'INV-P2-L2-1'] },
        ],
    },
    {
        name: 'Plant 3', location: 'Tamil Nadu, India', capacity: 200,
        blocks: [
            { name: 'Block A (L1)', inverters: ['INV-P3-L1-0', 'INV-P3-L1-1'] },
            { name: 'Block B (L2)', inverters: ['INV-P3-L2-0', 'INV-P3-L2-1'] },
        ],
    },
];

// ── Helper ───────────────────────────────────────────────────────────────────
function uuid() {
    return require('crypto').randomUUID();
}

async function main() {
    const db = await mysql.createConnection({
        host: process.env.DB_HOST || 'localhost',
        user: process.env.DB_USER || 'root',
        password: process.env.DB_PASSWORD || 'root',
        database: process.env.DB_NAME || 'Hackamined_new',
        multipleStatements: true,
    });

    console.log('✅ Connected to', process.env.DB_NAME || 'Hackamined_new');

    // ── 1. Admin ────────────────────────────────────────────────────────────
    const adminHash = await bcrypt.hash(ADMIN.pass, SALT_ROUNDS);
    const adminId = uuid();
    await db.execute(
        `INSERT IGNORE INTO admins (id, name, email, password_hash)
         VALUES (?, ?, ?, ?)`,
        [adminId, ADMIN.name, ADMIN.email, adminHash]
    );
    console.log(`👤 Admin created: ${ADMIN.email}`);

    // ── 2. Operator ─────────────────────────────────────────────────────────
    const opHash = await bcrypt.hash(OPERATOR.pass, SALT_ROUNDS);
    const opId = uuid();
    await db.execute(
        `INSERT IGNORE INTO operators (id, name, email, password_hash, created_by)
         VALUES (?, ?, ?, ?, ?)`,
        [opId, OPERATOR.name, OPERATOR.email, opHash, adminId]
    );
    console.log(`👷 Operator created: ${OPERATOR.email}`);

    // ── 3. Plants + Blocks + Inverters ──────────────────────────────────────
    for (const plant of PLANTS) {
        const plantId = uuid();
        await db.execute(
            `INSERT IGNORE INTO plants (id, name, location, total_capacity_kw, created_by)
             VALUES (?, ?, ?, ?, ?)`,
            [plantId, plant.name, plant.location, plant.capacity, adminId]
        );
        console.log(`🏭 Plant: ${plant.name}`);

        // Grant operator access to every plant
        await db.execute(
            `INSERT IGNORE INTO operator_plant_access (operator_id, plant_id)
             VALUES (?, ?)`,
            [opId, plantId]
        );

        for (const block of plant.blocks) {
            const blockId = uuid();
            await db.execute(
                `INSERT IGNORE INTO blocks (id, plant_id, name) VALUES (?, ?, ?)`,
                [blockId, plantId, block.name]
            );
            console.log(`  📦 Block: ${block.name}`);

            for (const invName of block.inverters) {
                await db.execute(
                    `INSERT IGNORE INTO inverters (id, block_id, name, serial_number, capacity_kw)
                     VALUES (UUID(), ?, ?, ?, 50.00)`,
                    [blockId, invName, `SN-${invName}`]
                );
                console.log(`    ⚡ Inverter: ${invName}`);
            }
        }
    }

    // ── 4. Default settings ──────────────────────────────────────────────────
    await db.execute(
        `INSERT IGNORE INTO settings (id, stale_timeout_minutes, thresholds)
         VALUES (1, 10, ?)`,
        [JSON.stringify({ A: [90, 100], B: [75, 89], C: [50, 74], D: [25, 49], E: [0, 24] })]
    );

    console.log('\n🎉 Seed complete!');
    console.log('─────────────────────────────────────');
    console.log(`Admin    : ${ADMIN.email}    / ${ADMIN.pass}`);
    console.log(`Operator : ${OPERATOR.email} / ${OPERATOR.pass}`);
    console.log('─────────────────────────────────────');

    await db.end();
}

main().catch(err => {
    console.error('❌ Seed failed:', err.message);
    process.exit(1);
});
