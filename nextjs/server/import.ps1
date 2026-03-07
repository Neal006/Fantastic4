$ErrorActionPreference = "Stop"

Write-Host "⏳ Importing database... Please wait..." -ForegroundColor Cyan
Write-Host ""

$dumpFile = Join-Path $PSScriptRoot "hackamined_dump.sql"
$mysqlPaths = @(
    "C:\xampp\mysql\bin\mysql.exe",
    "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
)

$mysqlExe = $null
foreach ($path in $mysqlPaths) {
    if (Test-Path $path) {
        $mysqlExe = $path
        Write-Host "✓ Found MySQL at: $path" -ForegroundColor Green
        break
    }
}

if (-not $mysqlExe) {
    Write-Host "❌ MySQL executable not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $dumpFile)) {
    Write-Host "❌ SQL dump file not found: $dumpFile" -ForegroundColor Red
    exit 1
}

$fileSize = [math]::Round((Get-Item $dumpFile).Length / 1MB, 2)
Write-Host "📦 File size: $fileSize MB" -ForegroundColor Yellow
Write-Host ""

$dbHost = if ($env:DB_HOST) { $env:DB_HOST } else { "localhost" }
$dbUser = if ($env:DB_USER) { $env:DB_USER } else { "root" }
$dbPass = if ($env:DB_PASSWORD) { $env:DB_PASSWORD } else { "root" }
$dbPort = if ($env:DB_PORT) { $env:DB_PORT } else { "3306" }

Write-Host "⚙️  Importing to MySQL..." -ForegroundColor Cyan
Write-Host "   Host: $dbHost" -ForegroundColor Gray
Write-Host "   User: $dbUser" -ForegroundColor Gray
Write-Host "   Port: $dbPort" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date

try {
    Get-Content $dumpFile -Raw | & $mysqlExe -h $dbHost -P $dbPort -u $dbUser "-p$dbPass" --max_allowed_packet=1G 2>&1 | Out-Null
    
    $duration = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)
    
    Write-Host ""
    Write-Host "✅ Success! Database imported in $duration seconds" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "❌ Import failed: $_" -ForegroundColor Red
    Write-Host ""
    exit 1
}
