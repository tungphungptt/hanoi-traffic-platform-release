# Download Google Drive data archive. Run from PowerShell.
$ErrorActionPreference = 'Stop'
$DriveId = '1xGaOlCwz7Zn7SbDZiWsfG0m2UGs7eqpO'
$OutDir = Join-Path $HOME 'Downloads'
$OutFile = Join-Path $OutDir 'aicc-data.zip'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (Get-Command aria2c -ErrorAction SilentlyContinue) {
    # Prefer resumable parallel download. The stable Drive URL avoids a session-specific `at=` token.
    $Url = "https://drive.google.com/uc?export=download&id=$DriveId"
    aria2c -c -x 16 -s 16 -k 1M --retry-wait=3 -m 20 -d $OutDir -o 'aicc-data.zip' $Url
    if ($LASTEXITCODE -eq 0 -and (Test-Path $OutFile)) {
        Write-Host "Downloaded: $OutFile"
        exit 0
    }
    Write-Warning 'aria2c did not finish successfully. Falling back to gdown.'
}

py -m pip install --upgrade gdown
if ($LASTEXITCODE -ne 0) { throw 'Could not install gdown.' }
py -m gdown --fuzzy "https://drive.google.com/file/d/$DriveId/view" -O $OutFile
if ($LASTEXITCODE -ne 0) { throw 'Download failed. Check that the Drive file is shared for anyone with the link.' }
Write-Host "Downloaded: $OutFile"
