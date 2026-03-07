const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
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
    console.log('📦 File size: ' + (fs.statSync(dumpPath).size / 1024 / 1024).toFixed(2) + ' MB');

    let connection;
    try {
        connection = await mysql.createConnection({
            host: DB_HOST,
            user: DB_USER,
            password: DB_PASS,
            port: DB_PORT
        });

        console.log('📝 Increasing max_allowed_packet...');
        await connection.query('SET GLOBAL max_allowed_packet=536870912');
        await connection.end();
        
        connection = await mysql.createConnection({
            host: DB_HOST,
            user: DB_USER,
            password: DB_PASS,
            port: DB_PORT
        });

        console.log('📖 Reading SQL dump file...');
        const sqlContent = fs.readFileSync(dumpPath, 'utf8');
        
        console.log('🔧 Parsing SQL statements...');
        const statements = [];
        let current = '';
        let inString = false;
        let stringDelimiter = '';
        let inComment = false;
        let lineComment = false;
        
        const lines = sqlContent.split('\n');
        let totalLines = lines.length;
        
        for (let lineNum = 0; lineNum < lines.length; lineNum++) {
            let line = lines[lineNum];
            
            if (lineNum % 10000 === 0 && lineNum > 0) {
                console.log(`   Parsing... ${((lineNum / totalLines) * 100).toFixed(1)}%`);
            }
            
            if (lineComment) {
                lineComment = false;
                continue;
            }
            
            if (line.trim().startsWith('--')) {
                continue;
            }
            
            for (let i = 0; i < line.length; i++) {
                const char = line[i];
                const nextChar = i < line.length - 1 ? line[i + 1] : '';
                const prevChar = i > 0 ? line[i - 1] : '';
                
                if (!inString && !inComment) {
                    if (char === '-' && nextChar === '-') {
                        lineComment = true;
                        break;
                    }
                    if (char === '/' && nextChar === '*') {
                        inComment = true;
                        current += char;
                        continue;
                    }
                }
                
                if (inComment) {
                    current += char;
                    if (char === '/' && prevChar === '*') {
                        inComment = false;
                    }
                    continue;
                }
                
                if ((char === '"' || char === "'" || char === '`') && prevChar !== '\\') {
                    if (!inString) {
                        inString = true;
                        stringDelimiter = char;
                    } else if (char === stringDelimiter) {
                        inString = false;
                        stringDelimiter = '';
                    }
                }
                
                current += char;
                
                if (char === ';' && !inString && !inComment) {
                    const stmt = current.trim();
                    if (stmt && !stmt.startsWith('--') && stmt.length > 5) {
                        statements.push(stmt);
                    }
                    current = '';
                }
            }
            
            current += '\n';
        }
        
        if (current.trim() && current.trim().length > 5) {
            statements.push(current.trim());
        }
        
        console.log(`✅ Parsed ${statements.length} SQL statements`);
        console.log('⚙️  Executing statements...\n');
        
        let successCount = 0;
        let skipCount = 0;
        
        for (let i = 0; i < statements.length; i++) {
            try {
                await connection.query(statements[i]);
                successCount++;
                
                if ((i + 1) % 50 === 0) {
                    console.log(`   ✓ Executed ${i + 1}/${statements.length} statements...`);
                }
            } catch (err) {
                if (err.message.includes('already exists') || 
                    err.message.includes('Duplicate entry') ||
                    err.message.includes('Unknown database')) {
                    skipCount++;
                } else {
                    console.warn(`   ⚠️  Statement ${i + 1}: ${err.message.substring(0, 80)}...`);
                    skipCount++;
                }
            }
        }
        
        console.log(`\n✅ Database import completed!`);
        console.log(`   Total statements: ${statements.length}`);
        console.log(`   Successfully executed: ${successCount}`);
        console.log(`   Skipped/Warnings: ${skipCount}`);
        
    } catch (error) {
        console.error('\n❌ Failed to import dump:', error.message);
        console.error('\n💡 Troubleshooting:');
        console.error('   1. Ensure MySQL server is running (XAMPP/WAMP)');
        console.error('   2. Check credentials in .env file');
        console.error('   3. Verify you have SUPER privilege for SET GLOBAL');
        process.exit(1);
    } finally {
        if (connection) {
            await connection.end();
        }
    }
}

importDatabase();
