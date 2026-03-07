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

        console.log('📝 Setting max_allowed_packet to 512MB...');
        await connection.query('SET GLOBAL max_allowed_packet=536870912');
        
        await connection.end();
        
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
        let successCount = 0;
        let inMultilineComment = false;
        let lineCount = 0;

        for await (const line of rl) {
            lineCount++;
            const trimmedLine = line.trim();

            if (trimmedLine.startsWith('/*') && !trimmedLine.startsWith('/*!')) {
                inMultilineComment = true;
            }
            if (inMultilineComment) {
                if (trimmedLine.includes('*/')) {
                    inMultilineComment = false;
                }
                continue;
            }

            if (!trimmedLine || trimmedLine.startsWith('--')) {
                continue;
            }

            currentStatement += (currentStatement ? ' ' : '') + line;

            if (trimmedLine.endsWith(';')) {
                const statement = currentStatement.trim();
                if (statement) {
                    try {
                        await connection.query(statement);
                        statementCount++;
                        successCount++;
                        if (statementCount % 50 === 0) {
                            console.log(`   ✓ Processed ${statementCount} statements (line ${lineCount})...`);
                        }
                    } catch (err) {
                        statementCount++;
                        if (!err.message.includes('already exists') && 
                            !err.message.includes('Duplicate entry') &&
                            !err.message.includes('Unknown database')) {
                            console.warn(`   ⚠️  Line ${lineCount}: ${err.message.substring(0, 60)}...`);
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
                successCount++;
            } catch (err) {
                if (!err.message.includes('already exists')) {
                    console.warn(`   ⚠️  Final statement: ${err.message.substring(0, 60)}...`);
                }
            }
        }

        console.log(`\n✅ Success! Database imported`);
        console.log(`   Total statements: ${statementCount}`);
        console.log(`   Successful: ${successCount}`);
        console.log(`   Warnings/Skipped: ${statementCount - successCount}`);
    } catch (error) {
        console.error('❌ Failed to import dump:', error.message);
        console.error('\n💡 Troubleshooting:');
        console.error('   1. Make sure MySQL server is running');
        console.error('   2. Check your credentials in .env file');
        console.error('   3. Ensure you have sufficient privileges');
        process.exit(1);
    } finally {
        if (connection) {
            await connection.end();
        }
    }
}

importDatabase();
