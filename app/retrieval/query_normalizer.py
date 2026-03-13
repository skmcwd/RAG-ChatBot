from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Pattern, TypedDict

from app.config import get_settings

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_ASCII_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")


class QueryNormalizationResult(TypedDict):
    """
    规范化结果结构。
    """

    raw_query: str
    normalized_query: str
    expanded_terms: list[str]
    guessed_categories: list[str]
    exact_terms: list[str]


@dataclass(frozen=True)
class SynonymConfig:
    """
    查询规范化所需的轻量配置。

    canonical_map:
        业务术语标准化映射，键为规范术语，值为其同义词/别名列表。
    category_rules:
        分类猜测规则，键为分类名，值为触发该分类的关键词列表。
    """

    canonical_map: dict[str, list[str]]
    category_rules: dict[str, list[str]]


@dataclass(frozen=True)
class _ReplacementRule:
    """
    内部替换规则。
    """

    canonical: str
    alias: str
    placeholder: str
    pattern: Pattern[str]


def _normalize_text(value: Any) -> str:
    """
    基础文本归一化：
    1. None 安全处理
    2. NFKC 归一化（全角/半角统一）
    3. 连续空白折叠
    4. 去除首尾空白
    """
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u3000", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _normalize_for_match(text: str) -> str:
    """
    用于匹配的文本标准化。
    在基础清洗上进一步做英文大小写标准化。
    """
    return _normalize_text(text).casefold()


def _lower_ascii_words(text: str) -> str:
    """
    将普通英文词统一为小写。
    规范术语（如 UKey）会在后续用占位符恢复，因此这里直接全量小写。
    """

    def repl(match: re.Match[str]) -> str:
        return match.group(0).lower()

    return _ASCII_WORD_RE.sub(repl, text)


def _unique_keep_order(items: list[str]) -> list[str]:
    """
    保持原顺序去重。
    """
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        text = _normalize_text(item)
        if not text:
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _is_ascii_like(text: str) -> bool:
    """
    判断术语是否主要由 ASCII 字母数字及常见连接符组成。
    """
    if not text:
        return False

    for ch in text:
        if ch.isascii():
            continue
        if ch.isspace():
            continue
        return False

    return True


def _compile_alias_pattern(alias: str) -> Pattern[str]:
    """
    将同义词别名编译为正则模式。

    规则：
    1. 空白使用 \\s+ 匹配，兼容多空格
    2. ASCII 术语使用边界限制，避免误替换单词子串
    3. 中文/混合术语直接做转义匹配
    """
    normalized_alias = _normalize_text(alias)
    if not normalized_alias:
        raise ValueError("alias 不能为空。")

    escaped = re.escape(normalized_alias)
    escaped = escaped.replace(r"\ ", r"\s+")

    if _is_ascii_like(normalized_alias):
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    else:
        pattern = escaped

    return re.compile(pattern, flags=re.IGNORECASE)


def _default_synonym_config() -> SynonymConfig:
    """
    内置默认词表。
    即使 synonyms.json 缺失，系统也能稳定运行。
    """
    canonical_map: dict[str, list[str]] = {
        "UKey": [
            "uk",
            "ukey",
            "u-key",
            "u key",
            "usbkey",
            "usb key",
            "usb-key",
            "key",
        ],
        "证书": [
            "ca证书",
            "uk证书",
            "证书初始化",
            "证书下载",
            "证书补发",
            "证书更新",
            "证书安装",
            "证书重置",
        ],
        "代发": [
            "代发工资",
            "工资代发",
            "批量代发",
            "批量发工资",
        ],
        "登录": [
            "登陆",
            "login",
            "sign in",
            "signin",
        ],
        "权限": [
            "授权",
            "复核",
            "审批",
            "操作员权限",
            "管理员权限",
        ],
        "回单": [
            "电子回单",
            "回执",
            "回单下载",
            "回单打印",
        ],
        "转账": [
            "付款",
            "汇款",
            "支付",
            "单笔转账",
            "批量转账",
        ],
        "控件": [
            "插件",
            "安全控件",
            "浏览器控件",
            "activex",
        ],
        "查询": [
            "流水",
            "明细",
            "余额",
            "对账",
            "记录查询",
        ],
        "账号权限": [
            "账户权限",
            "账号管理",
            "账户管理",
        ],
    }

    category_rules: dict[str, list[str]] = {
        "代发": ["代发", "代发工资", "批量代发", "工资代发"],
        "证书": ["证书", "ca", "证书下载", "证书初始化", "证书补发", "证书更新"],
        "UKey": ["ukey", "uk", "usbkey", "u-key", "usb key"],
        "登录": ["登录", "登陆", "密码", "用户名", "验证码", "认证"],
        "权限": ["权限", "授权", "复核", "审批", "管理员", "操作员"],
        "回单": ["回单", "回执", "电子回单", "红叉"],
        "转账": ["转账", "付款", "汇款", "支付"],
        "控件": ["控件", "插件", "activex", "浏览器控件", "安全控件"],
        "账号权限": ["账号权限", "账户权限", "账号管理", "账户管理"],
        "查询": ["查询", "流水", "明细", "余额", "对账"],
    }

    return SynonymConfig(
        canonical_map={k: _unique_keep_order(v) for k, v in canonical_map.items()},
        category_rules={k: _unique_keep_order(v) for k, v in category_rules.items()},
    )


def _merge_term_map(
        base: dict[str, list[str]],
        extra: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    合并术语映射并去重。
    """
    merged: dict[str, list[str]] = {key: list(values) for key, values in base.items()}

    for canonical, aliases in extra.items():
        existing = merged.get(canonical, [])
        merged[canonical] = _unique_keep_order([canonical, *existing, *aliases])

    for canonical, aliases in list(merged.items()):
        merged[canonical] = _unique_keep_order([canonical, *aliases])

    return merged


def _parse_synonym_json(data: dict[str, Any]) -> SynonymConfig:
    """
    解析 synonyms.json，兼容多种轻量结构。

    支持示例：
    1. 扁平结构
       {
         "UKey": ["uk", "usbkey"],
         "代发": ["代发工资", "批量代发"]
       }

    2. 显式结构
       {
         "canonical_map": {...},
         "category_rules": {...}
       }

    3. synonyms 别名结构
       {
         "synonyms": {...},
         "category_rules": {...}
       }
    """
    default_config = _default_synonym_config()

    if not data:
        return default_config

    if "canonical_map" in data or "synonyms" in data or "category_rules" in data:
        raw_canonical = data.get("canonical_map", data.get("synonyms", {}))
        raw_category_rules = data.get("category_rules", {})
    else:
        # 顶层扁平映射：全部视作 canonical_map
        raw_canonical = data
        raw_category_rules = {}

    canonical_map: dict[str, list[str]] = {}
    for canonical, aliases in raw_canonical.items():
        canonical_text = _normalize_text(canonical)
        if not canonical_text:
            continue

        alias_list: list[str]
        if isinstance(aliases, list):
            alias_list = [str(item) for item in aliases]
        elif isinstance(aliases, str):
            alias_list = [aliases]
        else:
            logger.warning("忽略非法同义词配置：canonical=%s", canonical)
            continue

        canonical_map[canonical_text] = _unique_keep_order([canonical_text, *alias_list])

    category_rules: dict[str, list[str]] = {}
    for category, keywords in raw_category_rules.items():
        category_text = _normalize_text(category)
        if not category_text:
            continue

        if isinstance(keywords, list):
            rule_values = [str(item) for item in keywords]
        elif isinstance(keywords, str):
            rule_values = [keywords]
        else:
            logger.warning("忽略非法分类规则配置：category=%s", category)
            continue

        category_rules[category_text] = _unique_keep_order(rule_values)

    merged_canonical_map = _merge_term_map(default_config.canonical_map, canonical_map)
    merged_category_rules = _merge_term_map(default_config.category_rules, category_rules)

    return SynonymConfig(
        canonical_map=merged_canonical_map,
        category_rules=merged_category_rules,
    )


def _get_default_synonyms_path() -> Path:
    """
    获取默认 synonyms.json 路径。
    """
    try:
        return get_settings().paths.synonyms_file
    except Exception:
        return Path(__file__).resolve().parent.parent.parent / "config" / "synonyms.json"


@lru_cache(maxsize=1)
def load_synonym_config(path: Path | None = None) -> SynonymConfig:
    """
    读取并缓存同义词配置。

    行为说明：
    1. 优先读取 config/synonyms.json
    2. 若文件不存在或格式非法，记录明确日志并回退到内置默认词表
    """
    target_path = path or _get_default_synonyms_path()

    if not target_path.exists():
        logger.warning("未找到 synonyms.json，已回退到内置默认词表：%s", target_path)
        return _default_synonym_config()

    if not target_path.is_file():
        logger.warning("synonyms.json 路径不是文件，已回退到内置默认词表：%s", target_path)
        return _default_synonym_config()

    try:
        content = target_path.read_text(encoding="utf-8")
        data = json.loads(content) if content.strip() else {}
    except OSError as exc:
        logger.warning("读取 synonyms.json 失败，已回退到内置默认词表：%s, err=%s", target_path, exc)
        return _default_synonym_config()
    except json.JSONDecodeError as exc:
        logger.warning("解析 synonyms.json 失败，已回退到内置默认词表：%s, err=%s", target_path, exc)
        return _default_synonym_config()

    if not isinstance(data, dict):
        logger.warning("synonyms.json 顶层结构不是对象，已回退到内置默认词表：%s", target_path)
        return _default_synonym_config()

    return _parse_synonym_json(data)


def _build_replacement_rules(config: SynonymConfig) -> list[_ReplacementRule]:
    """
    从词表构建替换规则，并按别名长度降序排序。
    长词优先，避免短词覆盖长词。
    """
    placeholder_map: dict[str, str] = {}
    rules: list[_ReplacementRule] = []
    placeholder_index = 0

    merged_map = _merge_term_map(config.canonical_map, {})

    for canonical, aliases in merged_map.items():
        if canonical not in placeholder_map:
            placeholder_map[canonical] = f"__CANONICAL_TERM_{placeholder_index}__"
            placeholder_index += 1

        placeholder = placeholder_map[canonical]
        for alias in aliases:
            normalized_alias = _normalize_text(alias)
            if not normalized_alias:
                continue

            rules.append(
                _ReplacementRule(
                    canonical=canonical,
                    alias=normalized_alias,
                    placeholder=placeholder,
                    pattern=_compile_alias_pattern(normalized_alias),
                )
            )

    rules.sort(key=lambda item: len(item.alias), reverse=True)
    return rules


def _replace_synonyms(
        text: str,
        config: SynonymConfig,
) -> tuple[str, list[str]]:
    """
    执行同义词替换。

    返回：
        replaced_text:
            用规范术语替换后的文本
        matched_canonicals:
            本次命中的规范术语列表
    """
    rules = _build_replacement_rules(config)
    working_text = text
    matched_canonicals: list[str] = []
    placeholder_to_canonical: dict[str, str] = {}

    for rule in rules:
        placeholder_to_canonical[rule.placeholder] = rule.canonical

        def repl(_: re.Match[str]) -> str:
            matched_canonicals.append(rule.canonical)
            return rule.placeholder

        working_text = rule.pattern.sub(repl, working_text)

    # 非规范术语英文统一转为小写
    working_text = _lower_ascii_words(working_text)

    # 恢复规范术语
    for placeholder, canonical in placeholder_to_canonical.items():
        working_text = working_text.replace(placeholder, canonical)

    working_text = _normalize_text(working_text)
    return working_text, _unique_keep_order(matched_canonicals)


def _guess_categories(
        raw_normalized: str,
        replaced_text: str,
        config: SynonymConfig,
) -> tuple[list[str], list[str]]:
    """
    依据规则猜测问题分类。

    返回：
        guessed_categories:
            猜测到的分类列表
        matched_triggers:
            命中的触发词列表，可用于 exact_terms
    """
    guessed_categories: list[str] = []
    matched_triggers: list[str] = []

    raw_match_text = _normalize_for_match(raw_normalized)
    replaced_match_text = _normalize_for_match(replaced_text)

    for category, keywords in config.category_rules.items():
        for keyword in keywords:
            normalized_keyword = _normalize_text(keyword)
            if not normalized_keyword:
                continue

            keyword_match = _normalize_for_match(normalized_keyword)
            if keyword_match and (
                    keyword_match in raw_match_text or keyword_match in replaced_match_text
            ):
                guessed_categories.append(category)
                matched_triggers.append(normalized_keyword)
                break

    return _unique_keep_order(guessed_categories), _unique_keep_order(matched_triggers)


def _build_expanded_terms(
        matched_canonicals: list[str],
        guessed_categories: list[str],
        config: SynonymConfig,
        exact_terms: list[str],
) -> list[str]:
    """
    生成扩展检索词。

    规则：
    1. 对已命中的规范术语，加入其别名
    2. 对已猜测分类，加入其规则关键词与同义词
    3. 剔除已在 exact_terms 中出现的词
    """
    candidates: list[str] = []
    seed_terms = _unique_keep_order([*matched_canonicals, *guessed_categories])

    for term in seed_terms:
        candidates.extend(config.canonical_map.get(term, []))
        candidates.extend(config.category_rules.get(term, []))

    exact_keys = {_normalize_text(item).casefold() for item in exact_terms}
    expanded: list[str] = []

    for term in _unique_keep_order(candidates):
        key = term.casefold()
        if key in exact_keys:
            continue
        expanded.append(term)

    return expanded


def normalize_query_with_config(
        query: str,
        config: SynonymConfig,
) -> QueryNormalizationResult:
    """
    纯函数式核心实现，便于单元测试。

    处理内容：
    1. 去除多余空白
    2. 全角半角归一化
    3. 英文大小写标准化
    4. 同义词替换
    5. 业务术语统一
    6. 简单规则分类
    """
    raw_query = "" if query is None else str(query)
    raw_normalized = _normalize_text(raw_query)

    if not raw_normalized:
        return QueryNormalizationResult(
            raw_query=raw_query,
            normalized_query="",
            expanded_terms=[],
            guessed_categories=[],
            exact_terms=[],
        )

    replaced_text, matched_canonicals = _replace_synonyms(raw_normalized, config)
    guessed_categories, matched_triggers = _guess_categories(
        raw_normalized=raw_normalized,
        replaced_text=replaced_text,
        config=config,
    )

    exact_terms = _unique_keep_order([*matched_canonicals, *matched_triggers])

    # 若归一化后直接包含分类名，也纳入 exact_terms
    lowered_normalized = _normalize_for_match(replaced_text)
    for category in guessed_categories:
        if _normalize_for_match(category) in lowered_normalized:
            exact_terms = _unique_keep_order([*exact_terms, category])

    expanded_terms = _build_expanded_terms(
        matched_canonicals=matched_canonicals,
        guessed_categories=guessed_categories,
        config=config,
        exact_terms=exact_terms,
    )

    return QueryNormalizationResult(
        raw_query=raw_query,
        normalized_query=replaced_text,
        expanded_terms=expanded_terms,
        guessed_categories=guessed_categories,
        exact_terms=exact_terms,
    )


def normalize_query(query: str) -> QueryNormalizationResult:
    """
    对用户问题做统一规范化。

    返回字段：
    - raw_query: 原始问题
    - normalized_query: 归一化后的问题
    - expanded_terms: 扩展检索词
    - guessed_categories: 猜测的业务分类
    - exact_terms: 精确命中的业务术语/触发词
    """
    config = load_synonym_config()
    return normalize_query_with_config(query=query, config=config)