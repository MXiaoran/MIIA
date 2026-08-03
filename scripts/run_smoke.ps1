$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

python -m miia.data smoke
python -m pytest
python -m miia.train --config configs/smoke.yaml --max-steps 2 --skip-validation
python -m miia.evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoints/last.pth --datasets rsicd rsitmd ucm --device cpu --report-dir reports/smoke
