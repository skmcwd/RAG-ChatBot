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


# 主程序：采用“放宽打包范围”的策略，优先保证稳定
for pkg in [
    "aiofiles",
    "annotated-doc",
    "annotated-types",
    "anyio",
    "asttokens",
    "attrs",
    "backcall",
    "bcrypt",
    "beautifulsoup4",
    "bleach",
    "brotli",
    "build",
    "certifi",
    "charset-normalizer",
    "chromadb",
    "click",
    "colorama",
    "decorator",
    "defusedxml",
    "distro",
    "docopt",
    "durationpy",
    "et_xmlfile",
    "executing",
    "fastapi",
    "fastjsonschema",
    "ffmpy",
    "filelock",
    "flatbuffers",
    "fsspec",
    "googleapis-common-protos",
    "gradio",
    "gradio_client",
    "groovy",
    "grpcio",
    "h11",
    "hf-xet",
    "httpcore",
    "httptools",
    "httpx",
    "huggingface_hub",
    "idna",
    "importlib_metadata",
    "importlib_resources",
    "ipython",
    "jedi",
    "Jinja2",
    "jiter",
    "jsonschema",
    "jsonschema-specifications",
    "jupyter_client",
    "jupyter_core",
    "jupyterlab_pygments",
    "kubernetes",
    "lxml",
    "markdown-it-py",
    "MarkupSafe",
    "matplotlib-inline",
    "mdurl",
    "mistune",
    "mmh3",
    "mpmath",
    "nbclient",
    "nbconvert",
    "nbformat",
    "numpy",
    "oauthlib",
    "onnxruntime",
    "openai",
    "openpyxl",
    "opentelemetry-api",
    "opentelemetry-exporter-otlp-proto-common",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-proto",
    "opentelemetry-sdk",
    "opentelemetry-semantic-conventions",
    "orjson",
    "overrides",
    "pandas",
    "pandocfilters",
    "parso",
    "pickleshare",
    "pillow",
    "pipreqs",
    "platformdirs",
    "prompt_toolkit",
    "protobuf",
    "pure_eval",
    "pybase64",
    "pydantic",
    "pydantic-settings",
    "pydantic_core",
    "pydub",
    "Pygments",
    "PyPika",
    "pyproject_hooks",
    "python-dateutil",
    "python-docx",
    "python-dotenv",
    "python-multipart",
    "python-pptx",
    "pytz",
    "PyYAML",
    "pyzmq",
    "rank-bm25",
    "RapidFuzz",
    "referencing",
    "requests",
    "requests-oauthlib",
    "rich",
    "rpds-py",
    "ruff",
    "safehttpx",
    "semantic-version",
    "setuptools",
    "shellingham",
    "six",
    "sniffio",
    "soupsieve",
    "stack-data",
    "starlette",
    "sympy",
    "tenacity",
    "tinycss2",
    "tokenizers",
    "tomlkit",
    "tornado",
    "tqdm",
    "traitlets",
    "typer",
    "typing-inspection",
    "typing_extensions",
    "tzdata",
    "urllib3",
    "uvicorn",
    "watchfiles",
    "wcwidth",
    "webencodings",
    "websocket-client",
    "websockets",
    "wheel",
    "xlsxwriter",
    "yarg",
    "zipp",
]:

    add_package(pkg)

# 收集你自己的项目模块，减少动态导入漏包风险
try:
    extend_unique(hiddenimports, collect_submodules("app"))
except Exception as exc:
    print(f"[WARN] collect_submodules(app) failed: {exc}")

try:
    extend_unique(hiddenimports, collect_submodules("scripts"))
except Exception as exc:
    print(f"[WARN] collect_submodules(scripts) failed: {exc}")


a = Analysis(
    ["main.py"],
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
    name="EviBank-RAG",
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
    name="EviBank-RAG",
)


