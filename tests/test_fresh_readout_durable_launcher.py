"""Windows-only harmless launcher checks; never register a campaign task."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Task Scheduler launcher")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_fresh_readout_durable.ps1"
POWERSHELL = str(
    Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
)


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(arguments, timeout=30):
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def without_filehash_command():
    """Keep other PS5 commands available while reproducing the missing hash function."""
    return (
        "$ErrorActionPreference='Stop';"
        "Import-Module Microsoft.PowerShell.Utility;"
        "Import-Module Microsoft.PowerShell.Management;"
        "Import-Module CimCmdlets;"
        "$PSModuleAutoLoadingPreference='None';"
        "Remove-Item Function:\\Get-FileHash -ErrorAction SilentlyContinue;"
        "if(Get-Command Get-FileHash -ErrorAction SilentlyContinue){throw 'Get-FileHash is available'};"
    )


def test_powershell_51_parser_accepts_launcher():
    command = (
        "$parseErrors=$null;"
        f"[System.Management.Automation.Language.Parser]::ParseFile({ps_quote(SCRIPT)},[ref]$null,[ref]$parseErrors)|Out-Null;"
        "if($parseErrors.Count){$parseErrors|Out-String;exit 1}"
    )
    result = run_ps(["-Command", command])
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["-ReviewedCampaign", "-LinuxRunnerPath", "bad;path"],
        ["-ReviewedCampaign", "-LinuxRunnerPath", "/home/safe.sh", "-Distribution", "Ubuntu --bad"],
        ["-ReviewedCampaign", "-LinuxRunnerPath", "/home/safe.sh", "-HfHome", '/home/bad"quote'],
    ],
)
def test_campaign_review_and_literal_argument_guards_precede_task_registration(tmp_path, extra):
    target = tmp_path / "must_not_exist"
    result = run_ps(
        ["-File", str(SCRIPT), "-Mode", "RegisterCampaign", "-RunDirectory", str(target), *extra]
    )
    assert result.returncode != 0
    assert not target.exists()


def test_foreground_process_drains_large_stderr_and_stdout_and_preserves_exit(tmp_path):
    child = tmp_path / "harmless_child.ps1"
    child.write_text(
        "$line='X'*1024;for($i=0;$i -lt 3000;$i++){[Console]::Out.WriteLine($line);[Console]::Error.WriteLine($line)};exit 7",
        encoding="utf-8",
    )
    driver = tmp_path / "driver.ps1"
    driver.write_text(
        "$ErrorActionPreference='Stop';"
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile({ps_quote(SCRIPT)},[ref]$null,[ref]$null);"
        "$names=@('Write-JsonFile','Utc-Now','Invoke-ForegroundProcess');"
        "$ast.FindAll({param($n)$n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $names -contains $n.Name},$true)|"
        "ForEach-Object {Invoke-Expression $_.Extent.Text};"
        f"$argsText='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File \"{child}\"';"
        f"$code=Invoke-ForegroundProcess {ps_quote(POWERSHELL)} $argsText {ps_quote(tmp_path)};"
        "@{exit_code=$code}|ConvertTo-Json",
        encoding="utf-8",
    )
    result = run_ps(["-File", str(driver)])
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["exit_code"] == 7
    assert (tmp_path / "wsl_stdout.log").read_bytes().count(b"X" * 1024) == 3000
    assert (tmp_path / "wsl_stderr.log").read_bytes().count(b"X" * 1024) == 3000


def test_worker_hashes_literal_script_without_get_filehash(tmp_path):
    """A direct harmless canary covers worker hashing without invoking WSL."""
    (tmp_path / "worker_config.json").write_text(
        json.dumps({"kind": "canary", "canary_seconds": 0}), encoding="utf-8"
    )
    command = (
        without_filehash_command()
        + f"& {ps_quote(SCRIPT)} -Mode Worker -RunDirectory {ps_quote(tmp_path)}"
    )
    result = run_ps(["-Command", command])
    assert result.returncode == 0, result.stderr + result.stdout
    started = json.loads((tmp_path / "started.json").read_text())
    assert started["worker_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "finished.json").read_text())["exit_code"] == 0
    heartbeat = json.loads((tmp_path / "heartbeat.jsonl").read_text())
    assert heartbeat["model_calls"] == 0


def test_harmless_wsl_worker_receives_environment_before_shell_execution(tmp_path):
    """Foreground local WSL printf only; no model calls, Python or scheduler task."""
    # Hosted Windows runners need not have this user's Ubuntu distribution.
    # Skip only absent registration; a registered but broken Ubuntu still fails.
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Lxss"
        ) as key:
            names = []
            for index in range(winreg.QueryInfoKey(key)[0]):
                with winreg.OpenKey(key, winreg.EnumKey(key, index)) as distro:
                    names.append(winreg.QueryValueEx(distro, "DistributionName")[0])
    except FileNotFoundError:
        pytest.skip("Local WSL integration requires a registered Ubuntu distribution")
    if "Ubuntu" not in names:
        pytest.skip("Local WSL integration requires a registered Ubuntu distribution")
    linux_runner = tmp_path / "harmless_runner.sh"
    linux_runner.write_bytes(
        b'#!/bin/bash\nset -eu\nprintf \'%s\\n\' "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS" "$HF_HOME" "$HF_HUB_CACHE" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$TOKENIZERS_PARALLELISM"\nprintf \'harmless stderr\\n\' >&2\n'
    )
    posix = linux_runner.as_posix()
    linux_path = "/mnt/" + posix[0].lower() + posix[2:]
    environment = {
        "OPENBLAS_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "HF_HOME": "/tmp/harmless cache path with spaces",
        "HF_HUB_CACHE": "/tmp/harmless-cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    config = {
        "kind": "campaign",
        "distribution": "Ubuntu",
        "linux_runner_path": linux_path,
        "linux_environment": environment,
    }
    (tmp_path / "worker_config.json").write_text(json.dumps(config), encoding="utf-8")
    result = run_ps(["-File", str(SCRIPT), "-Mode", "Worker", "-RunDirectory", str(tmp_path)])
    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "wsl_stdout.log").read_text().splitlines() == list(environment.values())
    assert (tmp_path / "wsl_stderr.log").read_text().strip() == "harmless stderr"
    assert json.loads((tmp_path / "finished.json").read_text())["exit_code"] == 0


def test_cim_no_window_worker_outlives_launcher_and_cleans_only_own_completed_canary(tmp_path):
    native = tmp_path / "exclusive_canary"
    command = (
        without_filehash_command()
        + f"& {ps_quote(SCRIPT)} -Mode Canary -LaunchMethod CimDetached "
        + f"-CanarySeconds 10 -RunDirectory {ps_quote(native)}"
    )
    result = run_ps(["-Command", command])
    assert result.returncode == 0, result.stderr + result.stdout
    registration = json.loads(result.stdout)
    assert registration["cim_return_value"] == 0
    assert registration["create_flags"] == 0x09000000
    assert registration["password_supplied"] is False
    assert registration["worker_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    # subprocess.run returned: the registering PowerShell process has exited.
    deadline = time.monotonic() + 25
    while not (native / "finished.json").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert (native / "finished.json").exists(), list(native.iterdir())
    finished = json.loads((native / "finished.json").read_text())
    assert finished["exit_code"] == 0
    started = json.loads((native / "started.json").read_text())
    assert started["worker_sha256"] == registration["worker_sha256"]
    assert (
        hashlib.sha256((native / "worker.ps1").read_bytes()).hexdigest()
        == registration["worker_sha256"]
    )
    assert any(p["name"].lower() == "wmiprvse.exe" for p in started["parent_chain"])
    heartbeats = [
        json.loads(line) for line in (native / "heartbeat.jsonl").read_text().splitlines()
    ]
    assert heartbeats[0]["sequence"] == 0
    assert heartbeats[-1]["elapsed_seconds"] >= 10
    assert all(h["model_calls"] == 0 for h in heartbeats)
    cleanup = run_ps(["-File", str(SCRIPT), "-Mode", "CleanupCanary", "-RunDirectory", str(native)])
    assert cleanup.returncode == 0, cleanup.stderr + cleanup.stdout
    assert json.loads((native / "task_cleanup.json").read_text())["scheduler_task_created"] is False
