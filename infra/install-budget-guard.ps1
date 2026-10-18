# The approved instance has a 20-hour ceiling. This task survives Codex closure.
$ErrorActionPreference = 'Stop'
$taskName = 'Forte-Brev-budget-stop-wd88a7vvt'
$cutoffUtc = [DateTimeOffset]::Parse('2026-10-18T06:01:25+00:00')
$guardArguments = '-d Ubuntu -- python3 /mnt/c/Users/Wezeso/Desktop/Forte/infra/budget_guard.py --execute'
$guardAction = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wsl.exe" -Argument $guardArguments
$guardTrigger = New-ScheduledTaskTrigger -Once -At $cutoffUtc.LocalDateTime
$guardSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RunOnlyIfNetworkAvailable -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 2) -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $taskName -Action $guardAction -Trigger $guardTrigger -Settings $guardSettings -Description 'Stop only the authorized Forte A100 at the 20-hour budget cutoff; data retained and storage cleanup remains required.' -Force | Out-Null
$guardTask = Get-ScheduledTask -TaskName $taskName
$guardInfo = Get-ScheduledTaskInfo -TaskName $taskName
[pscustomobject]@{Task=$guardTask.TaskName;State=$guardTask.State.ToString();NextRunTime=$guardInfo.NextRunTime.ToString('o');CutoffUtc=$cutoffUtc.ToString('o');WakeToRun=$guardTask.Settings.WakeToRun;ExecutionTimeLimit=$guardTask.Settings.ExecutionTimeLimit;RetryCount=$guardTask.Settings.RestartCount;RetryInterval=$guardTask.Settings.RestartInterval;RequiresComputerOnline=$true;DeletesData=$false} | ConvertTo-Json -Compress
