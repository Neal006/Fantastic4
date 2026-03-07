const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
const { Transform } = require('stream');
require('dotenv').config();

const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASS = process.env.DB_PASSWORD || 'root';
const DB_PORT = process.env.DB_PORT || 3306;

const dumpPath = path.join(__dirname, 'hackamined_dump.sql');

class SQLStatementParser extends Transform {
    constructor(options) {
        super(options);
        this.buffer = '';
        this.inString = false;
        this.stringChar = '';
        this.inComment = false;
    }

    _transform(chunk, encoding, callback) {
        this.buffer += chunk.toString();
        const statements = [];
        let current = '';
        
        for (let i = 0; i < this.buffer.length; i++) {
            const char = this.buffer[i];
            const prevChar = i > 0 ? this.buffer[i - 1] : '';
            const nextChar = i < this.buffer.length - 1 ? this.buffer[i + 1] : '';
            
            if (!this.inString && !this.inComment) {
                if (char === '/' && nextChar === '*' && !this.buffer.substring(i, i + 3).includes('!')) {
                    this.inComment = true;
                    current += char;
                    continue;
                }
                if (char === '-' && nextChar === '-') {
                    const lineEnd = this.buffer.indexOf('\n', i);
                    if (lineEnd !== -1) {
                        i = lineEnd;
                        continue;
                    }
                }
            }
            
            if (this.inComment) {
                current += char;
                if (char === '/' && prevChar === '*') {
                    this.inComment = false;
                }
                continue;
            }
            
            if ((char === '"' || char === "'" || char === '`') && prevChar !== '\\') {
                if (!this.inString) {
                    this.inString = true;
                    this.stringChar = char;
                } else if (char === this.stringChar) {
                    this.inString = false;
                }
            }
            
            current += char;
            
            if (char === ';' && !this.inString && !this.inComment) {
                const stmt = current.trim();
                if (stmt && stmt.length > 5 && !stmt.startsWith('--')) {
                    statements.push(stmt);
                    this.push(stmt + '\n__STATEMENT_END__\n');
                }
                current = '';
            }
        }
        
        this.buffer = current;
        callback();
    }

    _flush(callback) {
        if (this.buffer.trim() && this.buffer.trim().length > 5) {
            this.push(this.buffer.trim() + '\n__STATEMENT_END__\n');
        }
        callback();
    }
}

async function importDatabase() {
    if (!fs.existsSync(dumpPath)) {
        console.error('❌ Error: hackamined_dump.sql not found in the same folder.');
        process.exit(1);
    }

    const fileSize = (fs.statSync(dumpPath).size / 1024 / 1024).toFixed(2);
    console.log('⏳ Importing database... Please wait...');
    console.log(`📦 File size: ${fileSize} MB`);

    let connection;
    try {
        connection = await mysql.createConnection({
            host: DB_HOST,
            user: DB_USER,
            password: DB_PASS,
            port: DB_PORT
        });

        console.log('📝 Increasing max_allowed_packet to 512MB...');
        await connection.query('SET GLOBAL max_allowed_packet=536870912');
        await connection.end();
        
        connection = await mysql.createConnection({
            host: DB_HOST,
            user: DB_USER,
            password: DB_PASS,
            port: DB_PORT
        });

        console.log('⚙️  Processing SQL file in chunks...\n');
        
        const readStream = fs.createReadStream(dumpPath, { 
            encoding: 'utf8',
            highWaterMark: 1024 * 1024
        });
        
        const parser = new SQLStatementParser();
        readStream.pipe(parser);
        
        let statementBuffer = '';
        let statementCount = 0;
        let successCount = 0;
        let skipCount = 0;
        let lastUpdate = Date.now();
        
        for await (const chunk of parser) {
            statementBuffer += chunk;
            
            const statements = statementBuffer.split('\n__STATEMENT_END__\n');
            statementBuffer = statements.pop();
            
            for (const statement of statements) {
                if (!statement.trim()) continue;
                
                try {
                    await connection.query(statement);
                    successCount++;
                    statementCount++;
                    
                    if (Date.now() - lastUpdate > 2000) {
                        console.log(`   ✓ Processed ${statementCount} statements (${successCount} successful)...`);
                        lastUpdate = Date.now();
                    }
                } catch (err) {
                    statementCount++;
                    if (!err.message.includes('already exists') && 
                        !err.message.includes('Duplicate entry') &&
                        !err.message.includes('Unknown database')) {
                        if (skipCount < 5) {
                            console.warn(`   ⚠️  ${err.message.substring(0, 70)}...`);
                        }
                    }
                    skipCount++;
                }
            }
        }
        
        console.log(`\n✅ Database import completed!`);
        console.log(`   Total statements: ${statementCount}`);
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
