const mysql = require('mysql2/promise');
require('dotenv').config();

const pool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || 'root',
    database: process.env.DB_NAME || 'hackamined',
    port: process.env.DB_PORT || 3306,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
    timezone: '+00:00',
    // AWS RDS requires SSL/TLS for secure connections
    ssl: process.env.DB_SSL === 'true' ? {
        rejectUnauthorized: false // AWS RDS uses Amazon's CA, this allows connection
    } : undefined,
    // Connection timeout for cloud databases
    connectTimeout: 20000,
    // Enable multiple statements if needed
    multipleStatements: false,
});

// Test connection on startup
pool.getConnection()
    .then(conn => {
        console.log('✅ MySQL connected successfully');
        conn.release();
    })
    .catch(err => {
        console.error('❌ MySQL connection failed:', err.message);
        process.exit(1);
    });

module.exports = pool;
