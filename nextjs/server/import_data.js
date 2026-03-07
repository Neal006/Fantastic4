const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

// Configuration
const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASS = process.env.DB_PASSWORD || 'root';
const DB_PORT = process.env.DB_PORT || 3306;

const dumpPath = path.join(__dirname, 'hackamined_dump.sql');

if (!fs.existsSync(dumpPath)) {
    console.error('❌ Error: hackamined_dump.sql not found in the same folder.');
    process.exit(1);
}

console.log('⏳ Importing database... Please wait...');

// Build the shell command for the native MySQL CLI tool
// The CLI is much better at parsing big dumps than line-by-line buffers
const command = `mysql -h ${DB_HOST} -P ${DB_PORT} -u ${DB_USER} -p${DB_PASS} < "${dumpPath}"`;

exec(command, (error, stdout, stderr) => {
    if (error) {
        console.error('❌ Failed to import dump:', error.message);
        console.error('Do you have the "mysql" command installed on your machine?');
        return;
    }

    if (stderr && stderr.toLowerCase().includes('error')) {
        console.warn('⚠️ Import finished with warnings/errors:', stderr);
    } else {
        console.log('✅ Success! Database successfully imported.');
    }
});