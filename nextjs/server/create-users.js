/**
 * create-users.js
 * Creates admin and operator accounts with properly bcrypt-hashed passwords.
 * Run once: node create-users.js
 */
require('dotenv').config();
const mysql = require('mysql2/promise');
const bcrypt = require('bcryptjs');
const { randomUUID } = require('crypto');

const USERS = [
    // ── Admins ──────────────────────────────────────────────
    {
        table: 'admins',
        data: {
            id: randomUUID(),
            name: 'Admin User',
            email: 'admin@lumin.ai',
            password: 'Admin@123!',
        },
    },
    // ── Operators ───────────────────────────────────────────
    {
        table: 'operators',
        data: {
            id: randomUUID(),
            name: 'Arjun Mehta',
            email: 'arjun.mehta@lumin.ai',
            operator_id: 'OP001',
            password: 'Op@12345',
        },
    },
];

async function run() {
    const connection = await mysql.createConnection({
        host: process.env.DB_HOST || 'localhost',
        user: process.env.DB_USER || 'root',
        password: process.env.DB_PASSWORD || 'root',
        database: process.env.DB_NAME || 'hackamined',
    });

    console.log('✅ Connected to MySQL');

    for (const user of USERS) {
        const hash = await bcrypt.hash(user.data.password, 12);

        if (user.table === 'admins') {
            // Check if already exists
            const [existing] = await connection.query('SELECT id FROM admins WHERE email = ?', [user.data.email]);
            if (existing.length) {
                // Update the password hash
                await connection.query('UPDATE admins SET password_hash = ? WHERE email = ?', [hash, user.data.email]);
                console.log(`🔄 Updated admin: ${user.data.email}`);
            } else {
                await connection.query(
                    'INSERT INTO admins (id, name, email, password_hash) VALUES (?, ?, ?, ?)',
                    [user.data.id, user.data.name, user.data.email, hash]
                );
                console.log(`✅ Created admin: ${user.data.email}`);
            }
        } else {
            // Operator — login id is either email OR the operator_id field
            const [existing] = await connection.query('SELECT id FROM operators WHERE email = ?', [user.data.email]);
            if (existing.length) {
                await connection.query('UPDATE operators SET password_hash = ? WHERE email = ?', [hash, user.data.email]);
                console.log(`🔄 Updated operator: ${user.data.email} (login: ${user.data.operator_id})`);
            } else {
                // Check if operators table has operator_id column
                const [cols] = await connection.query('SHOW COLUMNS FROM operators LIKE "operator_id"');
                if (cols.length) {
                    await connection.query(
                        'INSERT INTO operators (id, name, email, operator_id, password_hash) VALUES (?, ?, ?, ?, ?)',
                        [user.data.id, user.data.name, user.data.email, user.data.operator_id, hash]
                    );
                } else {
                    await connection.query(
                        'INSERT INTO operators (id, name, email, password_hash) VALUES (?, ?, ?, ?)',
                        [user.data.id, user.data.name, user.data.email, hash]
                    );
                }
                console.log(`✅ Created operator: ${user.data.email} (login: ${user.data.operator_id})`);
            }
        }
    }

    // Summary
    console.log('\n── Summary ──────────────────────────────');
    const [admins] = await connection.query('SELECT name, email FROM admins');
    const [ops] = await connection.query('SELECT name, email FROM operators');
    console.log(`\nAdmins (${admins.length}):`);
    admins.forEach(a => console.log(`  • ${a.name} <${a.email}>`));
    console.log(`\nOperators (${ops.length}):`);
    ops.forEach(o => console.log(`  • ${o.name} <${o.email}>`));
    
    console.log('\n── Demo Credentials ─────────────────────');
    console.log('Admin:    admin@lumin.ai / Admin@123!');
    console.log('Operator: arjun.mehta@lumin.ai / Op@12345');
    console.log('\n🎉 Done! You can now log in with these credentials.');
    await connection.end();
}

run().catch(err => {
    console.error('❌ Error:', err.message);
    process.exit(1);
});
