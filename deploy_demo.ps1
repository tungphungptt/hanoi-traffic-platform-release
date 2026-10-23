[CmdletBinding()]
    param(
      [int]$Port = 8000,
      [switch]$SkipTests
    )
    $ErrorActionPreference = 'Stop'
    $Root = $PSScriptRoot
    Set-Location $Root
    Write-Host "`n=== HANOI TRAFFIC · LOCAL DEMO BOOTSTRAP ===" -ForegroundColor Cyan
    Write-Host "Project: $Root"

    if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
    $occupied = $null
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
      try { $occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 } catch { $occupied = $null }
      if ($occupied) {
        throw "Port $Port đang được sử dụng (PID $($occupied.OwningProcess)). Dừng server cũ bằng Ctrl+C hoặc chạy: .\deploy_demo.ps1 -Port 8001"
      }
    } else {
      $probe = $null
      try {
        $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $probe.Start()
        $probe.Stop()
      } catch {
        if ($probe) { try { $probe.Stop() } catch {} }
        throw "Port $Port có thể đang được sử dụng. Dừng server cũ hoặc chạy: .\deploy_demo.ps1 -Port 8001"
      }
    }

    # Robustly detect the interpreter; capture LASTEXITCODE immediately after the native process.
    # The generated .venv is project-owned and may be recreated if it is incompatible.
    function Get-PythonMinor {
      param([Parameter(Mandatory=$true)][string]$PythonExecutable)
      if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { return $null }
      $versionOutput = & $PythonExecutable -c "import sys; print('%s.%s' % (sys.version_info[0], sys.version_info[1]))" 2>$null
      $pythonExitCode = $LASTEXITCODE
      if ($pythonExitCode -ne 0) { return $null }
      $versionText = (@($versionOutput) -join "`n").Trim()
      $versionMatch = [regex]::Match($versionText, '(?m)^\s*(?<minor>3\.(?:11|12))\s*$')
      if ($versionMatch.Success) { return $versionMatch.Groups['minor'].Value }
      return $null
    }

    $VenvDir = Join-Path $Root '.venv'
    $VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
    $SupportedPythonVersions = @('3.12','3.11')
    $PythonMinor = Get-PythonMinor -PythonExecutable $VenvPython

    if ($PythonMinor -and -not ($SupportedPythonVersions -contains $PythonMinor)) {
      # Defensive guard if supported versions change in future; this branch is not expected
      # with the current regular expression, but never use an unapproved runtime silently.
      $PythonMinor = $null
    }

    if (-not $PythonMinor -and (Test-Path -LiteralPath $VenvDir)) {
      Write-Warning 'Virtual environment hiện tại không hợp lệ hoặc không thuộc Python 3.11/3.12; chỉ thư mục .venv được tạo lại.'
      Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
      $Launcher = Get-Command py -ErrorAction SilentlyContinue
      if (-not $Launcher) { throw 'Không tìm thấy Python Launcher (py). Cài Python 3.12 64-bit (hoặc 3.11) và bật Python Launcher.' }
      $Made = $false
      foreach ($Version in $SupportedPythonVersions) {
        if (Test-Path -LiteralPath $VenvDir) { Remove-Item -LiteralPath $VenvDir -Recurse -Force }
        Write-Host "  Trying Python $Version..." -ForegroundColor DarkCyan
        & $Launcher.Source ("-" + $Version) -m venv $VenvDir
        $createExitCode = $LASTEXITCODE
        $candidateMinor = if ($createExitCode -eq 0) { Get-PythonMinor -PythonExecutable $VenvPython } else { $null }
        if (($createExitCode -eq 0) -and (Test-Path -LiteralPath $VenvPython -PathType Leaf) -and ($candidateMinor -eq $Version)) {
          $PythonMinor = $candidateMinor
          $Made = $true
          break
        }
        if (Test-Path -LiteralPath $VenvDir) { Remove-Item -LiteralPath $VenvDir -Recurse -Force }
      }
      if (-not $Made) { throw 'Không tạo được venv bằng Python 3.12 hoặc 3.11. Cài Python 3.12 64-bit rồi thử lại.' }
    }

    # Always validate the final executable after environment creation/reuse.
    $PythonMinor = Get-PythonMinor -PythonExecutable $VenvPython
    if (-not ($SupportedPythonVersions -contains $PythonMinor)) {
      throw "Không xác minh được phiên bản Python của '$VenvPython'. Phiên bản cho phép: 3.11 hoặc 3.12."
    }
    Write-Host "Python runtime: $PythonMinor" -ForegroundColor DarkCyan

    Write-Host '[1/5] Installing app + test dependencies...' -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & $VenvPython -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Read the pip error above.' }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Package dependency consistency check failed. Resolve the pip conflicts above before launching.' }

    Write-Host '[2/5] Verifying bundled AICC model artifacts...' -ForegroundColor Yellow

    # Release must use model files included in this package.
    # Never copy weights from another local checkout.
    $BundleFiles = @('detector.onnx','predictor.onnx','scaler.pkl','manifest.json')
    $SourceModels = Join-Path $Root 'models'
    $TargetModels = Join-Path $Root 'models'

    $Missing = @(
      $BundleFiles | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $SourceModels $_) -PathType Leaf)
      }
    )
    $SourceComplete = ($Missing.Count -eq 0)

    if ($SourceComplete) {
      Write-Host '  Complete AICC bundle found inside release directory.' -ForegroundColor Green
      foreach ($Name in $BundleFiles) {
        $TargetFile = Join-Path $TargetModels $Name
        $Digest = (Get-FileHash -LiteralPath $TargetFile -Algorithm SHA256).Hash
        Write-Host ("  SHA256 {0}: {1}" -f $Name, $Digest) -ForegroundColor DarkGray
      }
    } else {
      throw "Release is missing bundled AICC files: $($Missing -join ', '). External model fallback is disabled."
    }

    if ($Missing.Count -gt 0) {
      throw "Incomplete AICC model bundle. Refusing to launch this release."
    }

    Write-Host '[Model smoke] Loading bundled ONNX models...' -ForegroundColor Yellow
    & $VenvPython tools/model_smoke.py
    if ($LASTEXITCODE -ne 0) {
      throw 'AICC model smoke failed. Check the bundled artifacts and feature manifest.'
    }

    # Dataset/videos are intentionally excluded from this release.
    # Do not copy data/input.mp4 from a previous workspace.    $env:PUBLIC_DEMO_MODE = 'false'
    $env:AICC_VIDEO_ENABLED = 'true'
    $env:AICC_SAMPLE_FPS = '6'

    Write-Host '[3/5] Checking release integrity and Python syntax...' -ForegroundColor Yellow
    & $VenvPython tools/release_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Release integrity check failed.' }
    & $VenvPython -m compileall -q app aicc traffic_brain tests tools
    if ($LASTEXITCODE -ne 0) { throw 'Python syntax check failed.' }
    if (-not $SkipTests) {
      Write-Host '[4/5] Running automated tests...' -ForegroundColor Yellow
      & $VenvPython -m pytest -q
      if ($LASTEXITCODE -ne 0) { throw 'Automated tests failed; deployment stopped.' }
    } else { Write-Host '[4/5] Test suite skipped by operator.' -ForegroundColor DarkYellow }

    Write-Host '[5/5] Starting local server...' -ForegroundColor Yellow
    $env:PYTHONUNBUFFERED = '1'
    $Proc = Start-Process -FilePath $VenvPython -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port',"$Port") -WorkingDirectory $Root -PassThru -WindowStyle Normal
    $BaseUrl = "http://127.0.0.1:$Port"
    $Ready = $false
    for ($i=0; $i -lt 45; $i++) {
      Start-Sleep -Seconds 1
      if ($Proc.HasExited) { throw "Uvicorn stopped unexpectedly (exit code $($Proc.ExitCode)). Check the server console." }
      try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 2
        if ($health.ok) { $Ready = $true; break }
      } catch {}
    }
    if (-not $Ready) { Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue; throw 'Server did not become healthy within 45 seconds.' }
    Write-Host "`nDEMO READY: $BaseUrl" -ForegroundColor Green
    Write-Host "API docs : $BaseUrl/docs"
    Write-Host "Health   : $BaseUrl/api/health"
    Write-Host "Server PID: $($Proc.Id) (separate console window)"
    Start-Process $BaseUrl
    Write-Host "`nDừng server: nhấn Ctrl+C trong cửa sổ Uvicorn. Local demo listens only on 127.0.0.1."
    Write-Host 'Lưu ý: điểm camera chưa có telemetry được đánh dấu là tham chiếu; clip local không phải feed RTSP trực tiếp. Dự báo +5/+10 chỉ xuất hiện khi có dữ liệu thời gian đủ điều kiện.' -ForegroundColor DarkYellow
