# Register the "ur5rl-watchdog" scheduled task: scripts/watchdog.py every 10 minutes while the user is logged on
# (no admin rights needed). Remove with: Unregister-ScheduledTask -TaskName ur5rl-watchdog -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $py)) { $py = Join-Path $root ".venv\Scripts\python.exe" }
$action = New-ScheduledTaskAction -Execute $py -Argument "`"$root\scripts\watchdog.py`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 10)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "ur5rl-watchdog" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "ur5rl-watchdog" | Select-Object TaskName, State
