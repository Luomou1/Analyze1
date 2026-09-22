from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app import update_checker


def test_update_uses_analyze1(monkeypatch, tmp_path: Path) -> None:
    requested = []

    def request(url: str) -> dict:
        requested.append(url)
        return {"tag_name": "v0.1.3", "assets": []}

    monkeypatch.setattr(update_checker, "_request_json", request)
    monkeypatch.setattr(update_checker, "update_cache_dir", lambda: tmp_path)
    info = update_checker.check_latest_update("0.1.2")
    assert requested == ["https://api.github.com/repos/Luomou1/Analyze1/releases/latest"]
    assert info.update_available


@pytest.mark.parametrize("scenario", ["success", "upload_failure", "digest_mismatch", "draft", "older_release"])
def test_release_cleanup_requires_verified_latest(tmp_path: Path, scenario: str) -> None:
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell 7 required for release script integration")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "upload_release_installer.ps1"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts" / script.name, script)
    installer = tmp_path / "dist" / "installer"
    installer.mkdir(parents=True)
    newest = installer / "AnalysisProgram-0.1.4-setup.exe"
    newest.write_bytes(b"new installer")
    old = installer / "DataAnalysis-0.1.3-setup.exe"
    old.write_bytes(b"old installer")
    portable = tmp_path / "dist" / "分析程序.exe"
    portable.write_bytes(b"portable")
    unrelated = installer / "other-tool.exe"
    unrelated.write_bytes(b"unrelated")
    # 在进程边界替代网络命令，真实执行发布脚本与文件删除逻辑。
    command = r'''
param($ScriptPath, $Scenario)
$ErrorActionPreference = 'Stop'
function python { $global:LASTEXITCODE = 0; '0.1.4' }
function gh {
    $global:LASTEXITCODE = 0
    if ($args[0] -eq 'release') {
        if ($Scenario -eq 'upload_failure') { $global:LASTEXITCODE = 1 }
        return
    }
    if ($args[1] -like '*/latest') {
        if ($Scenario -eq 'older_release') { 'v0.1.5' } else { 'v0.1.4' }
        return
    }
    $file = Get-Item 'dist/installer/AnalysisProgram-0.1.4-setup.exe'
    $digest = 'sha256:' + (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Scenario -in @('upload_failure', 'digest_mismatch')) { $digest = 'sha256:wrong' }
    @{
        draft = ($Scenario -eq 'draft'); prerelease = $false
        assets = @(@{name = $file.Name; size = $file.Length; digest = $digest})
    } | ConvertTo-Json -Depth 4
}
& $ScriptPath -Tag v0.1.4
'''
    harness = tmp_path / "harness.ps1"
    harness.write_text(command, encoding="utf-8")
    result = subprocess.run(
        [shell, "-NoProfile", "-File", str(harness), str(script), scenario],
        capture_output=True, encoding="utf-8", timeout=30,
    )
    assert newest.exists() and unrelated.exists()
    if scenario == "success":
        assert result.returncode == 0, result.stderr
        assert not old.exists() and not portable.exists()
    else:
        assert result.returncode != 0
        assert old.exists() and portable.exists()
