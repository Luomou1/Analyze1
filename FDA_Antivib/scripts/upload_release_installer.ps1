param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v\d+\.\d+\.\d+$')]
    [string]$Tag,
    [ValidateSet('Luomou1/Analyze1')]
    [string]$Repo = "Luomou1/Analyze1"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$version = python -c "from app import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0 -or $Tag -ne "v$version") { throw "Tag must match app version." }
$dist = Join-Path $Root "dist"
$source = Join-Path $dist "installer\分析程序-$version-setup.exe"
$assetName = "AnalysisProgram-$version-setup.exe"
$assetPath = Join-Path $dist "installer\$assetName"
if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $assetPath -Force }
if (-not (Test-Path -LiteralPath $assetPath)) { throw "Build the matching installer first." }

# 已上传同一工件时只核验，不重复传输大文件。
$hash = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
$before = gh api "repos/$Repo/releases/tags/$Tag"
if ($LASTEXITCODE -ne 0) { throw "Create the release first; local files kept." }
$existing = @(($before | ConvertFrom-Json).assets | Where-Object { $_.name -eq $assetName -and $_.digest -eq "sha256:$hash" })
if ($existing.Count -ne 1) {
    gh release upload $Tag $assetPath --repo $Repo --clobber
    if ($LASTEXITCODE -ne 0) { throw "Release upload failed; local files kept." }
}
$json = gh api "repos/$Repo/releases/tags/$Tag"
if ($LASTEXITCODE -ne 0) { throw "Cannot verify release; local files kept." }
$release = $json | ConvertFrom-Json
$asset = @($release.assets | Where-Object { $_.name -eq $assetName })
if ($release.draft -or $release.prerelease -or $asset.Count -ne 1 -or
    $asset[0].size -ne (Get-Item -LiteralPath $assetPath).Length -or
    $asset[0].digest -ne "sha256:$hash") {
    throw "Published asset verification failed; local files kept."
}
$latest = gh api "repos/$Repo/releases/latest" --jq '.tag_name'
if ($LASTEXITCODE -ne 0 -or $latest -ne $Tag) { throw "Not the latest release; local files kept." }

# 仅删除本项目生成的 EXE；不遍历工作区或依赖目录，不跟随链接。
$targets = @()
# 父级 build 也可能是目录联接，必须在枚举子目录前拒绝。
foreach ($parent in @($Root, $dist, (Join-Path $Root "build"))) {
    if ((Test-Path -LiteralPath $parent) -and
        ((Get-Item -LiteralPath $parent).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refuse linked output parent."
    }
}
foreach ($directory in @($dist, (Join-Path $dist "installer"), (Join-Path $Root "build\data_analysis"))) {
    if (-not (Test-Path -LiteralPath $directory)) { continue }
    $dirItem = Get-Item -LiteralPath $directory
    if ($dirItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refuse linked output directory." }
    $targets += Get-ChildItem -LiteralPath $directory -File -Filter '*.exe' | Where-Object {
        $_.FullName -ne $assetPath -and
        ($_.Name -in @('数据分析.exe', '分析程序.exe') -or $_.Name -match '^(数据分析|分析程序|DataAnalysis|AnalysisProgram)-\d+\.\d+\.\d+-setup\.exe$')
    }
}
foreach ($file in $targets) {
    if (-not $file.FullName.StartsWith($Root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Unsafe cleanup target." }
    Remove-Item -LiteralPath $file.FullName -Force
    Write-Host "Removed published build artifact: $($file.FullName)"
}
Write-Host "Verified release $Tag; retained $assetPath"
