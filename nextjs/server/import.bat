@echo off
setlocal enabledelayedexpansion

echo ⏳ Importing database... Please wait...
echo.

set "DUMP_FILE=%~dp0hackamined_dump_utf8.sql"
set "MYSQL_EXE="

if exist "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" (
    set "MYSQL_EXE=C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
    echo ✓ Found MySQL at: C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe
) else if exist "C:\xampp\mysql\bin\mysql.exe" (
    set "MYSQL_EXE=C:\xampp\mysql\bin\mysql.exe"
    echo ✓ Found MySQL at: C:\xampp\mysql\bin\mysql.exe
) else (
    echo ❌ MySQL executable not found
    exit /b 1
)

if not exist "%DUMP_FILE%" (
    echo ❌ SQL dump file not found: %DUMP_FILE%
    exit /b 1
)

echo.
echo ⚙️  Importing to MySQL...
echo    This may take a few minutes for large files...
echo.

if "%DB_HOST%"=="" set DB_HOST=localhost
if "%DB_USER%"=="" set DB_USER=root
if "%DB_PASSWORD%"=="" set DB_PASSWORD=root
if "%DB_PORT%"=="" set DB_PORT=3306
if "%DB_NAME%"=="" set DB_NAME=hackamined

echo 📝 Creating database if not exists...
"%MYSQL_EXE%" -h %DB_HOST% -P %DB_PORT% -u %DB_USER% -p%DB_PASSWORD% -e "CREATE DATABASE IF NOT EXISTS %DB_NAME%;" 2>nul

echo 📥 Importing SQL dump...
"%MYSQL_EXE%" -h %DB_HOST% -P %DB_PORT% -u %DB_USER% -p%DB_PASSWORD% --binary-mode=1 --max_allowed_packet=1G %DB_NAME% < "%DUMP_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Success! Database imported successfully
    echo.
) else (
    echo.
    echo ❌ Import failed with error code: %ERRORLEVEL%
    echo.
    echo 💡 Troubleshooting:
    echo    1. Make sure MySQL server is running
    echo    2. Check your credentials in .env file
    echo    3. Verify the database exists or will be created
    echo.
    exit /b 1
)
