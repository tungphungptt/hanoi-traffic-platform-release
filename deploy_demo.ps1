# Local launcher for Windows: creates .venv (Python 3.11/3.12), installs dependencies and opens the app.
param([int]$Port = 8000, [switch]$RunTests)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root
Write-Host "`n=== hanoiFlow · local ===" -ForegroundColor Cyan

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if (-not $py) { throw 'Không tìm thấy Python Launcher (py). Cài Python 3.12 64-bit rồi chạy lại.' }
  foreach ($v in '3.12', '3.11') {
    & $py.Source "-$v" -m venv .venv 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $VenvPython)) { break }
  }
  if (-not (Test-Path -LiteralPath $VenvPython)) { throw 'Không tạo được .venv bằng Python 3.12 hoặc 3.11.' }
}

Write-Host '[1/3] Cài thư viện...' -ForegroundColor Yellow
& $VenvPython -m pip install -q --upgrade pip
& $VenvPython -m pip install -q -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Cài thư viện thất bại; xem lỗi pip ở trên.' }
if (-not (Test-Path -LiteralPath (Join-Path $Root 'models\detector.onnx'))) { throw 'Thiếu models\detector.onnx.' }

if ($RunTests) {
  Write-Host '[2/3] Chạy kiểm thử...' -ForegroundColor Yellow
  & $VenvPython -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw 'Kiểm thử thất bại.' }
} else { Write-Host '[2/3] Bỏ qua kiểm thử (thêm -RunTests để chạy).' -ForegroundColor DarkGray }

Write-Host "[3/3] Khởi động http://127.0.0.1:$Port ..." -ForegroundColor Yellow
Start-Job -ScriptBlock { param($u) Start-Sleep 4; Start-Process $u } -ArgumentList "http://127.0.0.1:$Port" | Out-Null
& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port
