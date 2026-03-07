const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASS = process.env.DB_PASSWORD || 'root';
const DB_PORT = process.env.DB_PORT || 3306;
const DB_NAME = process.env.DB_NAME || 'hackamined';

const dumpPath = path.join(__dirname, 'hackamined_dump.sql');

function splitSqlStatements(sqlContent) {
    const statements = [];
    let currentStatement = '';
    let inString = false;
    let stringChar = '';
    
    for (let i = 0; i < sqlContent.length; i++) {
        const char = sqlContent[i];
        const prevChar = i > 0 ? sqlContent[i - 1] : '';
        
        if ((char === '"' || char === "'") && prevChar !== '\\') {
            if (!inString) {
                inString = true;
                stringChar = char;
            } else if (char === stringChar) {
                inString = false;
            }
        }
        
        currentStatement += char;
        
        if (char === ';' && !inString) {
            const trimmed = currentStatement.trim();
            if (trimmed && !trimmed.startsWith('--') && !trimmed.startsWith('/*')) {
                statements.push(trimmed);
            }
            currentStatement = '';
        }
    }
    
    if (currentStatement.trim()) {
        statements.push(currentStatement.trim());
    }
    
    return statements;
}

async function importDatabase() {
    if (!fs.existsSync(dumpPath)) {
        console.error('❌ Error: hackamined_dump.sql not found in the same folder.');
        process.exit(1);
    }

    console.log('⏳ Importing database... Please wait...');

    let connection;
    try {
        connection = await mysql.createConnection({
            host: DB_HOST,
            user: DB_USER,
            password: DB_PASS,
            port: DB_PORT
        });

        const sqlContent = fs.readFileSync(dumpPath, 'utf8');
        const statements = splitSqlStatements(sqlContent);
        
        console.log(`📝 Found ${statements.length} SQL statements to execute...`);
        
        for (let i = 0; i < statements.length; i++) {
            try {
                await connection.query(statements[i]);
                if ((i + 1) % 100 === 0) {
                    console.log(`   Processed ${i + 1}/${statements.length} statements...`);
                }
            } catch (err) {
                if (!err.message.includes('already exists')) {
                    console.warn(`⚠️  Warning on statement ${i + 1}: ${err.message.substring(0, 100)}`);
                }
            }
        }

        console.log('✅ Success! Database successfully imported.');
    } catch (error) {
        console.error('❌ Failed to import dump:', error.message);
        process.exit(1);
    } finally {
        if (connection) {
            await connection.end();
        }
    }
}

importDatabase();
