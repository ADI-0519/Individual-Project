# Windows service owns the worker process: CIM detached is the tested fallback
# to Task Scheduler. Never detach from an agent process with Start-Process.
# RegisterCampaign is for root review;
# only Canary is exercised during engineering validation.
[CmdletBinding()]
param(
    [ValidateSet('Canary', 'RegisterCampaign', 'Inspect', 'CleanupCanary', 'Worker')]
    [string]$Mode = 'Inspect',
    [string]$RunDirectory,
    [ValidateSet('TaskSchedulerConsole', 'CimDetached')][string]$LaunchMethod = 'CimDetached',
    [ValidateRange(10, 3600)][int]$CanarySeconds = 420,
    [string]$Distribution = 'Ubuntu',
    [string]$LinuxRunnerPath,
    [string]$HfHome = '/mnt/c/Users/adirj/OneDrive/Documents/GitHub/6CCSAPRJ - Individual project/Individual-Project/.cache/huggingface-wsl',
    [string]$HfHubCache = '/home/adirj/kcl-hf-cache',
    [switch]$ReviewedCampaign
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-JsonFile([string]$Path, $Value) {
    $encoded = $Value | ConvertTo-Json -Depth 20
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $encoded + [Environment]::NewLine, $utf8)
}

function Read-JsonFile([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Utc-Now { [DateTime]::UtcNow.ToString('o') }

function Get-LiteralFileSha256([string]$Path) {
    # Receipt hashes must not depend on Get-FileHash module discovery in PS5.
    $sha256 = [Security.Cryptography.SHA256]::Create()
    $stream = $null
    try {
        $stream = [IO.File]::OpenRead($Path)
        return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
        $sha256.Dispose()
    }
}

function Get-ProcessChain {
    $chain = @()
    $processId = $PID
    $seen = @{}
    for ($depth = 0; $depth -lt 12 -and $processId -gt 0; $depth++) {
        if ($seen.ContainsKey($processId)) { break }
        $seen[$processId] = $true
        $process = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$processId"
        if ($null -eq $process) { break }
        $chain += [PSCustomObject]@{
            pid = [int]$process.ProcessId
            parent_pid = [int]$process.ParentProcessId
            name = $process.Name
            executable_path = $process.ExecutablePath
            creation_date = $process.CreationDate.ToUniversalTime().ToString('o')
        }
        $processId = [int]$process.ParentProcessId
    }
    return $chain
}

function Scheduler-Service {
    $service = New-Object -ComObject 'Schedule.Service'
    $service.Connect()
    return $service
}

function Invoke-ForegroundProcess([string]$Executable, [string]$Arguments, [string]$Directory) {
    # Task Scheduler already owns this worker. This child is awaited, never
    # detached. Drain both streams asynchronously to avoid pipe deadlocks and
    # PowerShell 5's native-stderr error-record conversion.
    $outStream = [IO.File]::Open((Join-Path $Directory 'wsl_stdout.log'), [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $errStream = [IO.File]::Open((Join-Path $Directory 'wsl_stderr.log'), [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $process = New-Object Diagnostics.Process
    $process.StartInfo.FileName = $Executable
    $process.StartInfo.Arguments = $Arguments
    $process.StartInfo.WorkingDirectory = $Directory
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    try {
        [void]$process.Start()
        Write-JsonFile (Join-Path $Directory 'wsl_started.json') @{
            at = Utc-Now; pid = $process.Id; parent_pid = $PID
            executable = $Executable; arguments = $Arguments
        }
        $outCopy = $process.StandardOutput.BaseStream.CopyToAsync($outStream)
        $errCopy = $process.StandardError.BaseStream.CopyToAsync($errStream)
        while (-not $process.WaitForExit(5000)) {
            Write-JsonFile (Join-Path $Directory 'windows_worker_alive.json') @{
                at = Utc-Now; pid = $PID; wsl_pid = $process.Id; state = 'waiting_for_foreground_wsl'
            }
        }
        $process.WaitForExit()
        [void]$outCopy.GetAwaiter().GetResult()
        [void]$errCopy.GetAwaiter().GetResult()
        return [int]$process.ExitCode
    } finally {
        $outStream.Dispose()
        $errStream.Dispose()
        $process.Dispose()
    }
}

function Read-OwnRegistration([string]$Directory) {
    $registration = Read-JsonFile (Join-Path $Directory 'registration.json')
    if ($registration.schema -ne 'fresh-readout-durable-registration-v1' -or
        $registration.task_name -notmatch '^CodexFreshReadout(Canary|Campaign)_[0-9a-f]{32}$' -or
        $registration.user -ne [Security.Principal.WindowsIdentity]::GetCurrent().Name) {
        throw 'The receipt is not an owned current-user durable task.'
    }
    return $registration
}

if ($Mode -eq 'Worker') {
    $config = Read-JsonFile (Join-Path $RunDirectory 'worker_config.json')
    $started = @{
        schema = 'fresh-readout-durable-worker-v1'; started_at = Utc-Now
        pid = $PID; kind = $config.kind; user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        parent_chain = @(Get-ProcessChain)
        worker_sha256 = Get-LiteralFileSha256 $PSCommandPath
    }
    Write-JsonFile (Join-Path $RunDirectory 'started.json') $started
    try {
        if ($config.kind -eq 'canary') {
            $clock = [Diagnostics.Stopwatch]::StartNew()
            $sequence = 0
            $utf8 = New-Object System.Text.UTF8Encoding($false)
            while ($true) {
                $heartbeat = @{
                    at = Utc-Now; sequence = $sequence; elapsed_seconds = $clock.Elapsed.TotalSeconds
                    pid = $PID; kind = 'harmless_canary'; model_calls = 0
                } | ConvertTo-Json -Compress
                [IO.File]::AppendAllText((Join-Path $RunDirectory 'heartbeat.jsonl'), $heartbeat + [Environment]::NewLine, $utf8)
                if ($clock.Elapsed.TotalSeconds -ge [double]$config.canary_seconds) { break }
                $sequence++
                Start-Sleep -Seconds 5
            }
            $childExitCode = 0
        } elseif ($config.kind -eq 'campaign') {
            # Foreground WSL invocation: this Windows task remains alive until
            # the reviewed Linux recovery-and-completion runner exits.
            $wslExecutable = Join-Path $env:WINDIR 'System32/wsl.exe'
            if ($config.distribution -notmatch '^[A-Za-z0-9_.-]+$' -or
                $config.linux_runner_path -notmatch '^/[A-Za-z0-9_./-]+$') {
                throw 'Invalid literal WSL worker arguments.'
            }
            $argumentWords = @('--distribution', $config.distribution, '--exec', '/usr/bin/env')
            foreach ($item in $config.linux_environment.PSObject.Properties) {
                $argumentWords += $item.Name + '=' + [string]$item.Value
            }
            $argumentWords += @('/bin/bash', $config.linux_runner_path)
            $arguments = ($argumentWords | ForEach-Object {
                if ($_ -match '["\r\n\\]') { throw 'Unexpected quote, newline or backslash in literal Linux argument.' }
                if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
            }) -join ' '
            $childExitCode = Invoke-ForegroundProcess $wslExecutable $arguments $RunDirectory
        } else {
            throw 'Unknown worker kind.'
        }
        Write-JsonFile (Join-Path $RunDirectory 'finished.json') @{
            schema = 'fresh-readout-durable-worker-finish-v1'; finished_at = Utc-Now
            pid = $PID; kind = $config.kind; exit_code = $childExitCode
        }
        exit $childExitCode
    } catch {
        Write-JsonFile (Join-Path $RunDirectory 'failure.json') @{
            failed_at = Utc-Now; pid = $PID; message = $_.Exception.Message
        }
        exit 1
    }
}

if ($Mode -eq 'Inspect' -or $Mode -eq 'CleanupCanary') {
    if (-not $RunDirectory) { throw 'RunDirectory is required.' }
    $registration = Read-OwnRegistration $RunDirectory
    if ($registration.method -eq 'CimDetached') {
        $process = Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId=' + $registration.worker_pid)
        $startedPath = Join-Path $RunDirectory 'started.json'
        $sameProcess = $false
        if ($null -ne $process -and (Test-Path -LiteralPath $startedPath)) {
            $started = Read-JsonFile $startedPath
            $sameProcess = $process.CreationDate.ToUniversalTime().ToString('o') -eq $started.parent_chain[0].creation_date
        }
        $status = @{
            at = Utc-Now; method = 'CimDetached'; worker_pid = $registration.worker_pid
            worker_running = $sameProcess
            finished = Test-Path -LiteralPath (Join-Path $RunDirectory 'finished.json')
            failure = Test-Path -LiteralPath (Join-Path $RunDirectory 'failure.json')
        }
        Write-JsonFile (Join-Path $RunDirectory ('status_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff') + '.json')) $status
        if ($Mode -eq 'CleanupCanary') {
            if ($registration.kind -ne 'canary' -or $status.worker_running -or -not $status.finished) {
                throw 'Only an owned completed, nonrunning canary may be cleaned up.'
            }
            $finish = Read-JsonFile (Join-Path $RunDirectory 'finished.json')
            if ($finish.exit_code -ne 0) { throw 'Failed canary preserved.' }
            Write-JsonFile (Join-Path $RunDirectory 'task_cleanup.json') @{
                at = Utc-Now; method = 'CimDetached'; scheduler_task_created = $false
                cleanup = 'Worker exited; no scheduler task exists to delete; all native evidence retained.'
            }
        }
        $status | ConvertTo-Json -Depth 10
        exit 0
    }
    $service = Scheduler-Service
    $folder = $service.GetFolder('\')
    $task = $folder.GetTask($registration.task_name)
    $status = @{
        at = Utc-Now; task_name = $registration.task_name; state = [int]$task.State
        last_task_result = [int]$task.LastTaskResult
        last_run_time = $task.LastRunTime.ToUniversalTime().ToString('o')
        finished = Test-Path -LiteralPath (Join-Path $RunDirectory 'finished.json')
        failure = Test-Path -LiteralPath (Join-Path $RunDirectory 'failure.json')
    }
    Write-JsonFile (Join-Path $RunDirectory ('status_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff') + '.json')) $status
    if ($Mode -eq 'CleanupCanary') {
        if ($registration.kind -ne 'canary' -or $task.State -eq 4 -or -not $status.finished) {
            throw 'Only this launcher''s completed, nonrunning canary task can be removed.'
        }
        $finish = Read-JsonFile (Join-Path $RunDirectory 'finished.json')
        if ($finish.exit_code -ne 0) { throw 'Failed canary is preserved for inspection.' }
        $folder.DeleteTask($registration.task_name, 0)
        Write-JsonFile (Join-Path $RunDirectory 'task_cleanup.json') @{
            deleted_at = Utc-Now; task_name = $registration.task_name
            retained = 'All native heartbeat, registration, worker and status evidence retained.'
        }
    }
    $status | ConvertTo-Json -Depth 10
    exit 0
}

if ($Mode -eq 'RegisterCampaign' -and -not $ReviewedCampaign) {
    throw 'RegisterCampaign requires ReviewedCampaign after root reviews the Linux runner.'
}
if ($Mode -eq 'RegisterCampaign' -and
    ($LinuxRunnerPath -notmatch '^/[A-Za-z0-9_./-]+$' -or $Distribution -notmatch '^[A-Za-z0-9_.-]+$' -or
     $HfHome -notmatch '^/[A-Za-z0-9_./ -]+$' -or $HfHubCache -notmatch '^/[A-Za-z0-9_./ -]+$')) {
    throw 'Use an absolute Linux runner path and a literal distribution name.'
}

$kind = if ($Mode -eq 'Canary') { 'canary' } else { 'campaign' }
$tag = if ($kind -eq 'canary') { 'Canary' } else { 'Campaign' }
$identifier = [Guid]::NewGuid().ToString('N')
$taskName = 'CodexFreshReadout' + $tag + '_' + $identifier
$nativeBase = Join-Path $env:LOCALAPPDATA 'FreshReadoutDurable'
if (-not $RunDirectory) {
    $RunDirectory = Join-Path $nativeBase ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + $identifier)
}
$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
if (Test-Path -LiteralPath $RunDirectory) { throw 'Use a fresh exclusive RunDirectory.' }
[IO.Directory]::CreateDirectory($RunDirectory) | Out-Null
$workerPath = Join-Path $RunDirectory 'worker.ps1'
Copy-Item -LiteralPath $PSCommandPath -Destination $workerPath
$workerHash = Get-LiteralFileSha256 $workerPath
Write-JsonFile (Join-Path $RunDirectory 'worker_config.json') @{
    schema = 'fresh-readout-durable-worker-config-v1'; kind = $kind
    canary_seconds = $CanarySeconds; distribution = $Distribution; linux_runner_path = $LinuxRunnerPath
    linux_environment = [ordered]@{
        OPENBLAS_NUM_THREADS = '4'; OMP_NUM_THREADS = '4'; MKL_NUM_THREADS = '4'; NUMEXPR_NUM_THREADS = '4'
        HF_HOME = $HfHome; HF_HUB_CACHE = $HfHubCache
        HF_HUB_OFFLINE = '1'; TRANSFORMERS_OFFLINE = '1'; TOKENIZERS_PARALLELISM = 'false'
    }
}
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$registration = @{
    schema = 'fresh-readout-durable-registration-v1'; created_at = Utc-Now
    task_name = $taskName; kind = $kind; user = $user; worker_sha256 = $workerHash
    run_directory = $RunDirectory; caller_pid = $PID; caller_parent_chain = @(Get-ProcessChain)
    method = $LaunchMethod; logon_type = 3; run_level = 0
    action_window_style = 'Hidden'; password_supplied = $false
    new_language_model_calls = 0; status = 'registering'
}
try {
    if ($LaunchMethod -eq 'CimDetached') {
        $executable = Join-Path $env:WINDIR 'System32/WindowsPowerShell/v1.0/powershell.exe'
        $commandLine = '"' + $executable + '" -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $workerPath + '" -Mode Worker -RunDirectory "' + $RunDirectory + '"'
        $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
            ShowWindow = [uint16]0
            CreateFlags = [uint32]150994944 # CREATE_NO_WINDOW=0x08000000 | CREATE_BREAKAWAY_FROM_JOB=0x01000000
        }
        Write-JsonFile (Join-Path $RunDirectory 'registration.json') $registration
        $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = $commandLine; CurrentDirectory = $RunDirectory; ProcessStartupInformation = $startup
        }
        $registration.cim_return_value = [int]$created.ReturnValue
        if ($created.ReturnValue -ne 0) { throw ('Win32_Process.Create returned ' + $created.ReturnValue) }
        $registration.worker_pid = [int]$created.ProcessId
        $registration.create_flags = 150994944
        $registration.status = 'started'
        $registration.launch_returned_at = Utc-Now
        Write-JsonFile (Join-Path $RunDirectory 'registration.json') $registration
        $registration | ConvertTo-Json -Depth 20
        exit 0
    }
    $service = Scheduler-Service
    $folder = $service.GetFolder('\')
    $definition = $service.NewTask(0)
    $definition.RegistrationInfo.Description = 'Exclusive current-user FreshReadout durable ' + $kind + '; receipt ' + $RunDirectory
    $definition.RegistrationInfo.Author = $user
    $definition.Principal.UserId = $user
    $definition.Principal.LogonType = 3
    $definition.Principal.RunLevel = 0
    $definition.Settings.Enabled = $true
    $definition.Settings.Hidden = $true
    $definition.Settings.AllowDemandStart = $true
    $definition.Settings.MultipleInstances = 2
    $definition.Settings.DisallowStartIfOnBatteries = $false
    $definition.Settings.StopIfGoingOnBatteries = $false
    $definition.Settings.RunOnlyIfIdle = $false
    $definition.Settings.ExecutionTimeLimit = if ($kind -eq 'canary') { 'PT15M' } else { 'PT24H' }
    $action = $definition.Actions.Create(0)
    $action.Path = Join-Path $env:WINDIR 'System32/WindowsPowerShell/v1.0/powershell.exe'
    $action.Arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $workerPath + '" -Mode Worker -RunDirectory "' + $RunDirectory + '"'
    $action.WorkingDirectory = $RunDirectory
    # TASK_CREATE=2 never updates another task. Interactive token=3 requires the
    # current user to remain logged on; no password or elevated token is used.
    $task = $folder.RegisterTaskDefinition($taskName, $definition, 2, $user, $null, 3, $null)
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'registered_task.xml'), $task.Xml)
    $registration.status = 'registered'
    Write-JsonFile (Join-Path $RunDirectory 'registration.json') $registration
    $running = $task.Run($null)
    $registration.status = 'started'
    $registration.instance_guid = $running.InstanceGuid
    $registration.launch_returned_at = Utc-Now
    Write-JsonFile (Join-Path $RunDirectory 'registration.json') $registration
    $registration | ConvertTo-Json -Depth 20
} catch {
    $registration.status = 'registration_or_start_failed'
    $registration.error = $_.Exception.Message
    $registration.hresult = $_.Exception.HResult
    Write-JsonFile (Join-Path $RunDirectory 'registration.json') $registration
    throw
}
