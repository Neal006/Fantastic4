const mysql = require('mysql2');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASS = process.env.DB_PASSWORD || 'root';
const DB_PORT = process.env.DB_PORT || 3306;

const dumpPath = path.join(__dirname, 'hackamined_dump.sql');

console.log('⏳ Importing database... Please wait...');
console.log('📦 File size: ' + (fs.statSync(dumpPath).size / 1024 / 1024).toFixed(2) + ' MB\n');

const connection = mysql.createConnection({
    host: DB_HOST,
    user: DB_USER,
    password: DB_PASS,
    port: DB_PORT,
    multipleStatements: true,
    connectTimeout: 60000
});

connection.connect((err) => {
    if (err) {
        console.error('❌ Connection failed:', err.message);
        process.exit(1);
    }

    console.log('✓ Connected to MySQL');
    console.log('📝 Increasing packet size...');
    
    connection.query('SET GLOBAL max_allowed_packet=1073741824', (err) => {
        if (err) console.warn('⚠️  Could not increase packet size');

        console.log('📖 Reading SQL file...');
        const sql = fs.readFileSync(dumpPath, 'utf8');
        
        console.log('⚙️  Executing SQL (this will take a few minutes)...\n');
        
        const startTime = Date.now();
        connection.query(sql, (err, results) => {
            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            
            if (err) {
                console.error('❌ Import failed:', err.message);
                if (err.message.includes('max_allowed_packet')) {
                    console.error('\n💡 The SQL dump is too large for MySQL packet size.');
                    console.error('   Please use one of these alternatives:');
                    console.error('   1. MySQL Workbench: Server > Data Import');
                    console.error('   2. phpMyAdmin: Import tab');
                    console.error('   3. Add mysql.exe to your PATH and use: mysql -u root -p < hackamined_dump.sql');
                }
                connection.end();
                process.exit(1);
            }
            
            console.log(`✅ Success! Database imported in ${duration}s`);
            connection.end();
        });
    });
});
