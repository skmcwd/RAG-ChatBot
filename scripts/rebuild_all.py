from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

# 兼容直接执行：python scripts/rebuild_all.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts import parse_excel_faq as excel_parser_module  # noqa: E402
from scripts import parse_ppt_kb as ppt_parser_module  # noqa: E402
from scripts import parse_docx_manual as docx_parser_module  # noqa: E402
from scripts import build_indexes as build_indexes_module
from app.config import get_settings  # noqa: E402
from app.logging_utils import setup_logging  # noqa: E402
from app.models import KBChunk  # noqa: E402
from app.runtime import get_runtime_root  # noqa: E402
PARSER_MODULES = {
    "excel": excel_parser_module,
    "ppt": ppt_parser_module,
    "docx": docx_parser_module,
}
BUILD_INDEXES_MODULE = build_indexes_module
logger = logging.getLogger(__name__)

SUPPORTED_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
SUPPORTED_PPT_SUFFIXES = {".pptx"}
SUPPORTED_DOCX_SUFFIXES = {".docx"}

DEFAULT_BUILD_INDEX_BATCH_SIZE = 8
DEFAULT_SOURCE_CHUNKS_DIRNAME = "source_chunks"


class RebuildAllError(RuntimeError):
    """重建流程异常。"""


@dataclass
class RuntimePaths:
    root: Path
    raw_dir: Path
    parsed_dir: Path
    parsed_images_dir: Path
    index_dir: Path
    source_chunks_dir: Path
    kb_jsonl_path: Path
    rebuild_summary_path: Path


@dataclass
class DiscoveredFiles:
    excel_files: list[Path]
    ppt_files: list[Path]
    docx_files: list[Path]
    ignored_files: list[Path]


@dataclass
class ParseResult:
    source_type: str
    input_path: Path
    output_path: Path
    success: bool
    chunk_count: int
    error: str | None = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\u3000", " ").strip()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=True)


def _short_stable_name(path: Path, base_dir: Path) -> str:
    """
    为不同目录下的同名文件生成稳定且不冲突的输出名。
    """
    try:
        relative = path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        relative = path.resolve().as_posix()

    digest = hashlib.md5(relative.encode("utf-8")).hexdigest()[:8]
    stem = path.stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"{safe_stem}_{digest}"


def _iter_files(raw_dir: Path) -> Iterable[Path]:
    if not raw_dir.exists():
        return []
    return (
        p for p in raw_dir.rglob("*")
        if p.is_file()
    )


def discover_raw_files(raw_dir: Path) -> DiscoveredFiles:
    excel_files: list[Path] = []
    ppt_files: list[Path] = []
    docx_files: list[Path] = []
    ignored_files: list[Path] = []

    for path in _iter_files(raw_dir):
        name = path.name

        # 忽略 Office 临时文件，如 ~$xxx.xlsx
        if name.startswith("~$"):
            ignored_files.append(path)
            continue

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_EXCEL_SUFFIXES:
            excel_files.append(path)
        elif suffix in SUPPORTED_PPT_SUFFIXES:
            ppt_files.append(path)
        elif suffix in SUPPORTED_DOCX_SUFFIXES:
            docx_files.append(path)
        else:
            ignored_files.append(path)

    excel_files.sort()
    ppt_files.sort()
    docx_files.sort()
    ignored_files.sort()

    return DiscoveredFiles(
        excel_files=excel_files,
        ppt_files=ppt_files,
        docx_files=docx_files,
        ignored_files=ignored_files,
    )


def _get_paths() -> RuntimePaths:
    root = get_runtime_root()
    settings = get_settings()

    raw_dir = getattr(settings.paths, "raw_dir", root / "data" / "raw")
    parsed_dir = getattr(settings.paths, "parsed_dir", root / "data" / "parsed")
    parsed_images_dir = getattr(settings.paths, "parsed_images_dir", parsed_dir / "images")
    index_dir = getattr(settings.paths, "index_dir", root / "data" / "index")

    source_chunks_dir = parsed_dir / DEFAULT_SOURCE_CHUNKS_DIRNAME
    kb_jsonl_path = parsed_dir / "kb.jsonl"
    rebuild_summary_path = parsed_dir / "rebuild_summary.json"

    return RuntimePaths(
        root=root,
        raw_dir=Path(raw_dir),
        parsed_dir=Path(parsed_dir),
        parsed_images_dir=Path(parsed_images_dir),
        index_dir=Path(index_dir),
        source_chunks_dir=Path(source_chunks_dir),
        kb_jsonl_path=Path(kb_jsonl_path),
        rebuild_summary_path=Path(rebuild_summary_path),
    )


def _list_public_callables(module: Any) -> list[str]:
    """
    列出模块中所有公共可调用对象，便于定位实际暴露的解析函数名。
    """
    return sorted(
        name
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_")
    )


def clean_generated_artifacts(paths: RuntimePaths) -> None:
    """
    仅清理可重建产物，不动 raw 与配置文件。
    """
    logger.info("开始清理旧产物。")

    _ensure_dir(paths.parsed_dir)

    # 清理解析中间结果
    _remove_path(paths.source_chunks_dir)
    _ensure_dir(paths.source_chunks_dir)

    # 清理旧 kb
    if paths.kb_jsonl_path.exists():
        paths.kb_jsonl_path.unlink(missing_ok=True)

    # 清理解析出来的截图目录
    _remove_path(paths.parsed_images_dir)
    _ensure_dir(paths.parsed_images_dir)

    # 清理旧索引目录
    _remove_path(paths.index_dir)
    _ensure_dir(paths.index_dir)

    # 清理旧 summary
    if paths.rebuild_summary_path.exists():
        paths.rebuild_summary_path.unlink(missing_ok=True)

    logger.info("旧产物清理完成。")


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RebuildAllError(f"导入模块失败：{module_name}") from exc


def _get_first_callable(module: Any, candidates: list[str]) -> Callable[..., Any]:
    for name in candidates:
        func = getattr(module, name, None)
        if callable(func):
            return func

    available = _list_public_callables(module)
    raise RebuildAllError(
        f"模块 {getattr(module, '__name__', module)!r} 中未找到可调用函数。"
        f"候选={candidates}；实际可调用对象={available}"
    )


def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported_kwargs)


def _save_chunks_jsonl(chunks: list[KBChunk], output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk.model_dump(mode="json", exclude_none=False),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _load_chunks_jsonl(jsonl_path: Path) -> list[KBChunk]:
    if not jsonl_path.exists():
        raise RebuildAllError(f"中间 JSONL 不存在：{jsonl_path}")

    chunks: list[KBChunk] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
                chunk = KBChunk.model_validate(obj)
                chunks.append(chunk)
            except Exception as exc:
                raise RebuildAllError(
                    f"读取中间 JSONL 失败：file={jsonl_path}, line={line_no}"
                ) from exc
    return chunks


def _extract_chunk_candidate(item: Any) -> Any:
    """
    从不同解析器返回的对象中，尽量提取出可用于构造 KBChunk 的候选对象。

    支持：
    1. item 本身就是 KBChunk
    2. item 是 dict
    3. item 是 dict，且真正内容在 item["chunk"]
    4. item 有 .chunk 属性（如 ParsedSlideRecord）
    5. item 有 model_dump()，且 dump 后含 "chunk"
    6. item 是 dataclass，且 asdict 后含 "chunk"
    7. item 有 __dict__，且其中含 "chunk"
    """
    if item is None:
        return None

    # 1) 直接就是 KBChunk
    if isinstance(item, KBChunk):
        return item

    # 2) dict / dict["chunk"]
    if isinstance(item, dict):
        if "chunk" in item:
            return item["chunk"]
        return item

    # 3) 有 .chunk 属性
    chunk_attr = getattr(item, "chunk", None)
    if chunk_attr is not None:
        return chunk_attr

    # 4) pydantic / 自定义对象的 model_dump
    if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
        try:
            dumped = item.model_dump(mode="python")
        except TypeError:
            dumped = item.model_dump()
        if isinstance(dumped, dict):
            if "chunk" in dumped:
                return dumped["chunk"]
            return dumped

    # 5) dataclass
    if is_dataclass(item):
        dumped = asdict(item)
        if isinstance(dumped, dict):
            if "chunk" in dumped:
                return dumped["chunk"]
            return dumped

    # 6) 普通对象 __dict__
    if hasattr(item, "__dict__"):
        dumped = {
            k: v
            for k, v in vars(item).items()
            if not str(k).startswith("_")
        }
        if dumped:
            if "chunk" in dumped:
                return dumped["chunk"]
            return dumped

    return item


def _coerce_to_kbchunk(
        item: Any,
        *,
        module_name: str,
        input_path: Path,
        index: int,
) -> KBChunk:
    """
    将解析器返回的任意条目安全转换为 KBChunk。
    """
    candidate = _extract_chunk_candidate(item)

    try:
        if isinstance(candidate, KBChunk):
            return candidate
        return KBChunk.model_validate(candidate)
    except Exception as exc:
        raise RebuildAllError(
            f"解析结果中存在无效 KBChunk：module={module_name}, "
            f"input={input_path}, index={index}, "
            f"item_type={type(item).__name__}, candidate_type={type(candidate).__name__}"
        ) from exc


def _parse_with_module(
        *,
        module: Any,
        source_type: str,
        input_path: Path,
        output_path: Path,
        parsed_images_dir: Path,
        parse_candidates: list[str],
) -> ParseResult:
    """
    通用解析器封装：
    1. 动态导入脚本模块
    2. 查找 parse_* 函数
    3. 查找 save_chunks_to_jsonl / fallback 自己写 JSONL
    """
    logger.info("开始解析 %s 文件：%s", source_type, input_path)

    try:
        module_name = getattr(module, "__name__", str(module))
        parse_func = _get_first_callable(module, parse_candidates)

        # # 常见保存函数名
        # save_func = getattr(module, "save_chunks_to_jsonl", None)
        # if save_func is not None and not callable(save_func):
        #     save_func = None

        kwargs = {
            "input_path": input_path,
            "sheet_name": 0,
            "image_dir": parsed_images_dir,
            "images_output_dir": parsed_images_dir,
            "image_output_dir": parsed_images_dir,
            "output_image_dir": parsed_images_dir,
            "parsed_images_dir": parsed_images_dir,
            "images_dir": parsed_images_dir,
        }

        chunks_obj = _call_with_supported_kwargs(parse_func, **kwargs)

        if not isinstance(chunks_obj, list):
            raise RebuildAllError(
                f"{module_name} 返回结果不是 list：input={input_path}"
            )

        # chunks: list[KBChunk] = []
        # for idx, item in enumerate(chunks_obj, start=1):
        #     try:
        #         if isinstance(item, KBChunk):
        #             chunks.append(item)
        #         else:
        #             chunks.append(KBChunk.model_validate(item))
        #     except Exception as exc:
        #         raise RebuildAllError(
        #             f"解析结果中存在无效 KBChunk：module={module_name}, input={input_path}, index={idx}"
        #         ) from exc

        # _ensure_dir(output_path.parent)
        # if save_func is not None:
        #     _call_with_supported_kwargs(save_func, chunks=chunks, output_path=output_path)
        # else:
        #     _save_chunks_jsonl(chunks, output_path)

        logger.debug(
            "%s 解析返回类型：list_len=%s, first_item_type=%s",
            source_type,
            len(chunks_obj),
            type(chunks_obj[0]).__name__ if chunks_obj else "empty",
        )

        chunks: list[KBChunk] = []
        for idx, item in enumerate(chunks_obj, start=1):
            chunk = _coerce_to_kbchunk(
                item,
                module_name=module_name,
                input_path=input_path,
                index=idx,
            )
            chunks.append(chunk)

        # 统一由 rebuild_all 自己写标准 JSONL，避免依赖各脚本保存函数的输入类型
        _ensure_dir(output_path.parent)
        _save_chunks_jsonl(chunks, output_path)
        logger.info(
            "%s 解析完成：input=%s, output=%s, chunks=%s",
            source_type,
            input_path,
            output_path,
            len(chunks),
        )
        return ParseResult(
            source_type=source_type,
            input_path=input_path,
            output_path=output_path,
            success=True,
            chunk_count=len(chunks),
            error=None,
        )

    except Exception as exc:
        logger.exception("%s 解析失败：input=%s, err=%s", source_type, input_path, exc)
        return ParseResult(
            source_type=source_type,
            input_path=input_path,
            output_path=output_path,
            success=False,
            chunk_count=0,
            error=str(exc),
        )


def parse_all_sources(paths: RuntimePaths, discovered: DiscoveredFiles) -> list[ParseResult]:
    results: list[ParseResult] = []

    # Excel
    excel_out_dir = paths.source_chunks_dir / "excel"
    _ensure_dir(excel_out_dir)
    for file_path in discovered.excel_files:
        output_name = _short_stable_name(file_path, paths.raw_dir) + ".jsonl"
        results.append(
            _parse_with_module(
                # module_name="scripts.parse_excel_faq",
                module=PARSER_MODULES["excel"],
                source_type="excel",
                input_path=file_path,
                output_path=excel_out_dir / output_name,
                parsed_images_dir=paths.parsed_images_dir,
                parse_candidates=["parse_excel_faq"]
            )
        )

    # PPT
    ppt_out_dir = paths.source_chunks_dir / "ppt"
    _ensure_dir(ppt_out_dir)
    for file_path in discovered.ppt_files:
        output_name = _short_stable_name(file_path, paths.raw_dir) + ".jsonl"
        results.append(
            _parse_with_module(
                # module_name="scripts.parse_ppt_kb",
                module=PARSER_MODULES["ppt"],
                source_type="ppt",
                input_path=file_path,
                output_path=ppt_out_dir / output_name,
                parsed_images_dir=paths.parsed_images_dir,
                parse_candidates=["parse_ppt_to_records"]
            )
        )

    # DOCX
    docx_out_dir = paths.source_chunks_dir / "docx"
    _ensure_dir(docx_out_dir)
    for file_path in discovered.docx_files:
        output_name = _short_stable_name(file_path, paths.raw_dir) + ".jsonl"
        results.append(
            _parse_with_module(
                # module_name="scripts.parse_docx_manual",
                module=PARSER_MODULES["docx"],
                source_type="docx",
                input_path=file_path,
                output_path=docx_out_dir / output_name,
                parsed_images_dir=paths.parsed_images_dir,
                parse_candidates=["parse_docx_manual"]
            )
        )

    return results


def merge_chunks_from_parse_results(parse_results: list[ParseResult]) -> list[KBChunk]:
    """
    合并所有中间 JSONL，并按 chunk_hash / doc_id 去重。
    """
    all_chunks: list[KBChunk] = []

    for result in parse_results:
        if not result.success:
            continue
        chunks = _load_chunks_jsonl(result.output_path)
        all_chunks.extend(chunks)

    deduped: list[KBChunk] = []
    seen: set[str] = set()

    for chunk in all_chunks:
        key = _normalize_text(chunk.chunk_hash) or _normalize_text(chunk.doc_id)
        if not key:
            # 极端兜底：理论上 KBChunk 不应走到这里
            key = hashlib.md5(
                json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()

        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    deduped.sort(
        key=lambda x: (
            _normalize_text(x.source_type),
            _normalize_text(x.source_file),
            _normalize_text(x.title),
            _normalize_text(x.doc_id),
        )
    )
    return deduped


def write_kb_jsonl(chunks: list[KBChunk], output_path: Path) -> None:
    _save_chunks_jsonl(chunks, output_path)
    logger.info("统一知识库已写出：%s（共 %s 条）", output_path, len(chunks))


@contextmanager
def _patched_argv(argv: list[str]):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


# def rebuild_indexes(
#         kb_jsonl_path: Path,
#         *,
#         batch_size: int,
#         collection_name: str | None,
#         log_level: str,
# ) -> int:
#     """
#     直接复用 scripts.build_indexes.py 的 CLI 入口，避免重复实现索引逻辑。
#     """
#     module = _import_module("scripts.build_indexes")
#     main_func = getattr(module, "main", None)
#     if not callable(main_func):
#         raise RebuildAllError("scripts.build_indexes 中不存在可调用的 main()。")
#
#     argv = [
#         "build_indexes.py",
#         "--input",
#         str(kb_jsonl_path),
#         "--rebuild",
#         "--batch-size",
#         str(batch_size),
#         "--log-level",
#         log_level,
#     ]
#     if collection_name:
#         argv.extend(["--collection-name", collection_name])
#
#     logger.info("开始重建索引：argv=%s", argv)
#
#     with _patched_argv(argv):
#         return_code = int(main_func())
#
#     if return_code != 0:
#         raise RebuildAllError(f"build_indexes.py 执行失败，返回码={return_code}")
#
#     logger.info("索引重建完成。")
#     return return_code
def rebuild_indexes(
        kb_jsonl_path: Path,
        *,
        batch_size: int,
        collection_name: str | None,
        log_level: str,
) -> int:
    """
    直接复用 scripts.build_indexes.py 的 CLI 入口，避免重复实现索引逻辑。
    这里使用静态导入的模块对象，避免打包后动态导入 scripts.build_indexes 失败。
    """
    module = BUILD_INDEXES_MODULE
    module_name = getattr(module, "__name__", "scripts.build_indexes")

    main_func = getattr(module, "main", None)
    if not callable(main_func):
        raise RebuildAllError(f"{module_name} 中不存在可调用的 main()。")

    argv = [
        "build_indexes.py",
        "--input",
        str(kb_jsonl_path),
        "--rebuild",
        "--batch-size",
        str(batch_size),
        "--log-level",
        log_level,
    ]
    if collection_name:
        argv.extend(["--collection-name", collection_name])

    logger.info("开始重建索引：argv=%s", argv)

    with _patched_argv(argv):
        return_code = int(main_func())

    if return_code != 0:
        raise RebuildAllError(f"build_indexes.py 执行失败，返回码={return_code}")

    logger.info("索引重建完成。")
    return return_code

def write_summary(
        *,
        output_path: Path,
        paths: RuntimePaths,
        discovered: DiscoveredFiles,
        parse_results: list[ParseResult],
        final_chunk_count: int,
) -> None:
    summary = {
        "root": str(paths.root),
        "raw_dir": str(paths.raw_dir),
        "parsed_dir": str(paths.parsed_dir),
        "index_dir": str(paths.index_dir),
        "discovered": {
            "excel_files": [str(p) for p in discovered.excel_files],
            "ppt_files": [str(p) for p in discovered.ppt_files],
            "docx_files": [str(p) for p in discovered.docx_files],
            "ignored_files": [str(p) for p in discovered.ignored_files],
        },
        "parse_results": [
            {
                "source_type": r.source_type,
                "input_path": str(r.input_path),
                "output_path": str(r.output_path),
                "success": r.success,
                "chunk_count": r.chunk_count,
                "error": r.error,
            }
            for r in parse_results
        ],
        "final_chunk_count": final_chunk_count,
    }

    _ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("重建摘要已写出：%s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键重建企业网银 FAQ RAG 的知识库与索引。"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，例如 DEBUG / INFO / WARNING / ERROR",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BUILD_INDEX_BATCH_SIZE,
        help=f"传递给 build_indexes.py 的 batch size，默认 {DEFAULT_BUILD_INDEX_BATCH_SIZE}",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="可选：覆盖 build_indexes.py 的 collection 名称",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="是否跳过旧产物清理。默认会清理 parsed/images/source_chunks/index 等目录。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, module_name=__name__)

    if args.batch_size <= 0:
        logger.error("--batch-size 必须大于 0。")
        return 1

    try:
        paths = _get_paths()
        logger.info(
            "开始执行一键重建：root=%s, raw_dir=%s, parsed_dir=%s, index_dir=%s",
            paths.root,
            paths.raw_dir,
            paths.parsed_dir,
            paths.index_dir,
        )

        if not paths.raw_dir.exists():
            raise RebuildAllError(f"raw 目录不存在：{paths.raw_dir}")

        if not args.skip_clean:
            clean_generated_artifacts(paths)
        else:
            logger.warning("已启用 --skip-clean，将保留旧产物目录。")

        discovered = discover_raw_files(paths.raw_dir)
        logger.info(
            "扫描完成：excel=%s, ppt=%s, docx=%s, ignored=%s",
            len(discovered.excel_files),
            len(discovered.ppt_files),
            len(discovered.docx_files),
            len(discovered.ignored_files),
        )

        total_supported = (
                len(discovered.excel_files)
                + len(discovered.ppt_files)
                + len(discovered.docx_files)
        )
        if total_supported == 0:
            raise RebuildAllError(
                f"raw 目录下未发现可处理文件：{paths.raw_dir}。支持类型：xlsx/xls/xlsm、pptx、docx。"
            )

        parse_results = parse_all_sources(paths, discovered)

        success_count = sum(1 for r in parse_results if r.success)
        fail_count = sum(1 for r in parse_results if not r.success)
        logger.info("解析阶段完成：成功文件=%s, 失败文件=%s", success_count, fail_count)

        chunks = merge_chunks_from_parse_results(parse_results)
        if not chunks:
            raise RebuildAllError("所有文件均未生成有效知识块，无法继续构建索引。")

        write_kb_jsonl(chunks, paths.kb_jsonl_path)

        rebuild_indexes(
            paths.kb_jsonl_path,
            batch_size=args.batch_size,
            collection_name=args.collection_name,
            log_level=args.log_level,
        )

        write_summary(
            output_path=paths.rebuild_summary_path,
            paths=paths,
            discovered=discovered,
            parse_results=parse_results,
            final_chunk_count=len(chunks),
        )

        logger.info("一键重建完成：最终知识块数=%s", len(chunks))
        return 0

    except Exception as exc:
        logger.exception("一键重建失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
