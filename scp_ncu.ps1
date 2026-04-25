param(
    [string]$Source = "smartcho:/home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit/ex.ncu-rep",
    [string]$Destination = ".\$(Get-Date -Format 'yyMMdd_HHmm').ncu-rep"
)

scp $Source $Destination
Write-Host "Copied $Source -> $Destination"
