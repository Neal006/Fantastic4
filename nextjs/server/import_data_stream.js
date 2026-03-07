const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
require('dotenv').config();

const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASS = process.env.DB_PASSWORD || 'root';
const DB_PORT = process.env.DB_PORT || 3306;

const dumpPath = path.join(__dirname, 'hackamined_dump.sql');

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

        const fileStream = fs.createReadStream(dumpPath, { encoding: 'utf8' });
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });

        let currentStatement = '';
        let statementCount = 0;
        let inMultilineComment = false;

        for await (const line of rl) {
            const trimmedLine = line.trim();

            if (trimmedLine.startsWith('/*')) {
                inMultilineComment = true;
            }
            if (inMultilineComment) {
                if (trimmedLine.endsWith('*/')) {
                    inMultilineComment = false;
                }
                continue;
            }

            if (!trimmedLine || trimmedLine.startsWith('--')) {
                continue;
            }

            currentStatement += ' ' + line;

            if (trimmedLine.endsWith(';')) {
                const statement = currentStatement.trim();
                if (statement) {
                    try {
                        await connection.query(statement);
                        statementCount++;
                        if (statementCount % 100 === 0) {
                            console.log(`   Processed ${statementCount} statements...`);
                        }
                    } catch (err) {
                        if (!err.message.includes('already exists') && !err.message.includes('Duplicate entry')) {
                            console.warn(`⚠️  Warning: ${err.message.substring(0, 80)}`);
                        }
                    }
                }
                currentStatement = '';
            }
        }

        if (currentStatement.trim()) {
            try {
                await connection.query(currentStatement.trim());
                statementCount++;
            } catch (err) {
                console.warn(`⚠️  Warning on final statement: ${err.message.substring(0, 80)}`);
            }
        }

        console.log(`✅ Success! Database imported (${statementCount} statements executed).`);
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
