# Prepared for explicit user approval before invocation. Binds only the new test host.
$ErrorActionPreference = 'Stop'
$taskName = 'Forte-Brev-new-budget-stop-plp5me7vz'
$arguments = '-d Ubuntu -- python3 /mnt/c/Users/Wezeso/Desktop/Forte/infra/new_account_budget_guard.py --instance-name forte-new-h100 --instance-id plp5me7vz --organization daniyarber0307-9589d1-kqod --started-at 2026-10-18T04:20:00Z --hourly-usd 3 --budget-usd 50 --reserve-usd 2 --cutoff-utc 2026-10-18T12:55:00Z --execute'
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wsl.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Once -At ([DateTimeOffset]::Parse('2026-10-18T17:55:00+05:00').LocalDateTime)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'Task already exists; inspect it before replacing.'
}
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description 'Stop only Forte new-account H100 before the approved 18:00 local deadline.' | Out-Null
Get-ScheduledTask -TaskName $taskName | ForEach-Object {
    [pscustomobject]@{ name=$_.TaskName;state=[string]$_.State;enabled=$_.Settings.Enabled;trigger=$_.Triggers.StartBoundary }
} | ConvertTo-Json -Compress
