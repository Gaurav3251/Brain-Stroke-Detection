param(
    [string]$VenvPath = ".venv"
)

python -m venv $VenvPath
Write-Host "Virtual environment created at $VenvPath"
Write-Host "Activate with: $VenvPath\Scripts\Activate.ps1"
