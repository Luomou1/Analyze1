# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


ROOT = Path(SPECPATH).resolve().parent


hiddenimports = [
    "finufft",
    "matplotlib.backends.backend_qtagg",
    "pyvistaqt",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[],
    binaries=collect_dynamic_libs("finufft"),
    datas=[(str(ROOT / "assets" / "app_icon.ico"), "assets")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib.tests",
        "numba",
        "pandas",
        "pytest",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
# Qt 6.11 使用 Windows 系统 ICU 的无版本函数名。PATH 中 Poppler 的
# 同名 icuuc.dll 仅导出带版本后缀的函数，误打包会导致 QtGui 加载失败。
# Windows 10/11 自带系统 ICU，不能用第三方同名 DLL 覆盖它。
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() != "icuuc.dll"]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="分析程序",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "app_icon.ico"),
)
