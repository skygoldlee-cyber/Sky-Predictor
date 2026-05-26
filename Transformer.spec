# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from PyInstaller.utils.hooks import collect_submodules


def _safe_collect_submodules(name: str) -> list[str]:
    try:
        return collect_submodules(name)
    except Exception:
        return []


block_cipher = None

# App icon
APP_ICON_PATH = "assets/beacon.ico"

hiddenimports: list[str] = []
hiddenimports += _safe_collect_submodules("qasync")
hiddenimports += _safe_collect_submodules("PySide6")
hiddenimports += _safe_collect_submodules("numpy")
hiddenimports += _safe_collect_submodules("pandas")


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[(APP_ICON_PATH, ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Transformer",
    debug=False,
    icon=APP_ICON_PATH,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
