const mysql = require('mysql2');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASS = process.env.DB_PASSWORD || 'root';
const DB_PORT = process.env.DB_PORT || 3306;

const dumpPath = path.join(__dirname, 'hackamined_dump.sql');

function importDatabase() {
    if (!fs.existsSync(dumpPath)) {
        console.error('❌ Error: hackamined_dump.sql not found in the same folder.');
        process.exit(1);
    }

    const fileSize = (fs.statSync(dumpPath).size / 1024 / 1024).toFixed(2);
    console.log('⏳ Importing database... Please wait...');
    console.log(`📦 File size: ${fileSize} MB`);

    const connection = mysql.createConnection({
        host: DB_HOST,
        user: DB_USER,
        password: DB_PASS,
        port: DB_PORT,
        multipleStatements: true
    });

    connection.connect((err) => {
        if (err) {
            console.error('❌ Failed to connect to MySQL:', err.message);
            console.error('\n💡 Make sure MySQL is running and credentials are correct');
            process.exit(1);
        }

        console.log('📝 Setting max_allowed_packet...');
        connection.query('SET GLOBAL max_allowed_packet=536870912', (err) => {
            if (err) {
                console.warn('⚠️  Could not set max_allowed_packet:', err.message);
            }

            console.log('📖 Reading and executing SQL dump...');
            console.log('   This may take several minutes for large files...\n');

            const stream = fs.createReadStream(dumpPath, { encoding: 'utf8' });
            let buffer = '';
            let chunkSize = 0;
            const maxChunkSize = 16 * 1024 * 1024; // 16MB chunks
            let totalProcessed = 0;
            let pendingQueries = 0;
            let completed = false;

            const executeChunk = (chunk, callback) => {
                if (!chunk.trim()) {
                    callback();
                    return;
                }

                pendingQueries++;
                connection.query(chunk, (err, results) => {
                    pendingQueries--;
                    if (err && !err.message.includes('already exists') && 
                        !err.message.includes('Duplicate entry')) {
                        console.warn(`   ⚠️  ${err.message.substring(0, 70)}...`);
                    } else {
                        totalProcessed++;
                        if (totalProcessed % 10 === 0) {
                            console.log(`   ✓ Processed ${totalProcessed} chunks...`);
                        }
                    }
                    callback();
                });
            };

            const processBuffer = (callback) => {
                if (buffer.length === 0) {
                    callback();
                    return;
                }

                const lastSemicolon = buffer.lastIndexOf(';');
                if (lastSemicolon === -1) {
                    callback();
                    return;
                }

                const toExecute = buffer.substring(0, lastSemicolon + 1);
                buffer = buffer.substring(lastSemicolon + 1);

                executeChunk(toExecute, callback);
            };

            stream.on('data', (chunk) => {
                buffer += chunk;
                chunkSize += chunk.length;

                if (chunkSize >= maxChunkSize) {
                    stream.pause();
                    processBuffer(() => {
                        chunkSize = 0;
                        stream.resume();
                    });
                }
            });

            stream.on('end', () => {
                processBuffer(() => {
                    if (buffer.trim()) {
                        executeChunk(buffer, () => {
                            completed = true;
                            checkCompletion();
                        });
                    } else {
                        completed = true;
                        checkCompletion();
                    }
                });
            });

            stream.on('error', (err) => {
                console.error('❌ Error reading file:', err.message);
                connection.end();
                process.exit(1);
            });

            const checkCompletion = () => {
                if (completed && pendingQueries === 0) {
                    console.log(`\n✅ Database import completed!`);
                    console.log(`   Processed ${totalProcessed} chunks`);
                    connection.end();
                } else if (completed) {
                    setTimeout(checkCompletion, 100);
                }
            };
        });
    });
}

importDatabase();
