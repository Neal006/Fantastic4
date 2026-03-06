const fs = require('fs');
const path = require('path');
const mysql = require('mysql2/promise');
require('dotenv').config();

async function setup() {
    const connection = await mysql.createConnection({
        host: process.env.DB_HOST || 'localhost',
        user: process.env.DB_USER || 'root',
        password: process.env.DB_PASSWORD || 'root',
        database: process.env.DB_NAME || 'hackamined',
        multipleStatements: true,
    });

    console.log('✅ Connected to MySQL');

    const schemaSQL = fs.readFileSync(path.join(__dirname, 'db', 'schema.sql'), 'utf8');
    const seedSQL = fs.readFileSync(path.join(__dirname, 'db', 'seed.sql'), 'utf8');

    console.log('📋 Running schema...');
    await connection.query(schemaSQL);
    console.log('✅ Schema created');

    console.log('🌱 Running seed...');
    await connection.query(seedSQL);
    console.log('✅ Seed data inserted');

    const [rows] = await connection.query('SELECT COUNT(*) AS count FROM inverters');
    console.log(`\n📊 Setup complete! Inverters in DB: ${rows[0].count}`);
    const [admins] = await connection.query('SELECT name, email FROM admins');
    console.log('👤 Admin:', admins[0]);
    const [ops] = await connection.query('SELECT name, email FROM operators');
    console.log('👥 Operators:', ops.map((o) => o.name).join(', '));

    await connection.end();
    console.log('\n🎉 Database setup complete! Run "npm run dev" to start the server.');
}

setup().catch(err => {
    console.error('❌ Setup failed:', err.message);
    process.exit(1);
});
