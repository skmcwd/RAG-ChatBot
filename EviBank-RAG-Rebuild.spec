# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

project_root = Path(__name__).resolve().parent

datas = []
binaries = []
hiddenimports = []


def extend_unique(target, items):
    for item in items:
        if item not in target:
            target.append(item)


def add_package(pkg_name: str):
    try:
        d, b, h = collect_all(pkg_name)
        extend_unique(datas, d)
        extend_unique(binaries, b)
        extend_unique(hiddenimports, h)
    except Exception as exc:
        print(f"[WARN] collect_all({pkg_name}) failed: {exc}")

    try:
        extend_unique(datas, collect_data_files(pkg_name, include_py_files=True))
    except Exception as exc:
        print(f"[WARN] collect_data_files({pkg_name}) failed: {exc}")

    try:
        extend_unique(datas, copy_metadata(pkg_name))
    except Exception as exc:
        print(f"[WARN] copy_metadata({pkg_name}) failed: {exc}")

    try:
        extend_unique(hiddenimports, collect_submodules(pkg_name))
    except Exception as exc:
        print(f"[WARN] collect_submodules({pkg_name}) failed: {exc}")


# 重建程序同样采用“放宽打包范围”的策略，换稳定性
for pkg in [
    "gradio",
    "gradio_client",
    "safehttpx",
    "httpx",
    "httpcore",
    "anyio",
    "sniffio",
    "openai",
    "pandas",
    "numpy",
    "chromadb",
    "openpyxl",
    "pptx",
    "docx",
    "PIL",
    "yaml",
    "orjson",
    "pydantic",
    "pydantic_settings",
]:
    add_package(pkg)

try:
    extend_unique(hiddenimports, collect_submodules("app"))
except Exception as exc:
    print(f"[WARN] collect_submodules(app) failed: {exc}")

try:
    extend_unique(hiddenimports, [
        "scripts",
        "scripts.parse_excel_faq",
        "scripts.parse_ppt_kb",
        "scripts.parse_docx_manual",
        "scripts.build_indexes",
    ])
except Exception as exc:
    print(f"[WARN] collect_submodules(scripts) failed: {exc}")

a = Analysis(
    ["scripts/rebuild_all.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["sqlite3"],
    hookspath=[str(project_root / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EviBank-RAG-Rebuild",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="EviBank-RAG-Rebuild",
)