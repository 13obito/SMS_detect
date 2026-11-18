# Run from any directory. Use an activated Python environment with requirements installed.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python run_experiments.py midsemester --download
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed. Read the Python error above.' }
    Write-Host 'Finished. Open results\midsemester\report_fill_in.txt.'
}
finally {
    Pop-Location
}
