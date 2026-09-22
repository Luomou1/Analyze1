param([switch]$SkipInstaller)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$version = python -c "from app import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0) { throw "Cannot read app version." }
python scripts\generate_app_icon.py
if ($LASTEXITCODE -ne 0) { throw "Icon generation failed." }
python -m PyInstaller --clean --noconfirm packaging\data_analysis.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

# 实际运行打包结果，避免仅源码测试通过而分发包缺失或混入错误 DLL。
$smoke = Start-Process -FilePath (Join-Path $Root "dist\分析程序.exe") -ArgumentList '--smoke-test-imports' -WindowStyle Hidden -PassThru
if (-not $smoke.WaitForExit(60000)) { throw "Packaged EXE smoke test timed out; installer not built." }
if ($smoke.ExitCode -ne 0) { throw "Packaged EXE smoke test failed; installer not built." }

if (-not $SkipInstaller) {
    $iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    $isccPath = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if ($null -ne $iscc) { $isccPath = $iscc.Source }
    if (-not (Test-Path -LiteralPath $isccPath)) {
        throw "Inno Setup ISCC.exe not found; only the portable EXE was built."
    }
    & $isccPath "/DMyAppVersion=$version" packaging\data_analysis.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed." }
}
# 旧产物只在远端发布成功并核对 SHA-256 后清理，构建失败不触发删除。
