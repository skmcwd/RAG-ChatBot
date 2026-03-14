from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from app.config import AppConfigError, get_settings
from app.logging_utils import setup_logging
from app.services.chat_service import ChatService
import app.ui.app as ui_app_module

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_SHARE = False
DEFAULT_INBROWSER = False


class AppStartupError(RuntimeError):
    """应用启动阶段异常。"""


def _parse_bool(value: Any, default: bool = False) -> bool:
    """
    将常见布尔表示稳健转换为 bool。
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    return default


def _parse_int(value: Any, default: int) -> int:
    """
    将值稳健转换为 int。
    """
    if value is None:
        return default

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _load_raw_yaml_config(settings_file: Path) -> dict[str, Any]:
    """
    读取原始 settings.yaml。

    这里不依赖 app.config.Settings 的字段定义，
    便于在不修改主配置模型的情况下读取可选的 server 启动参数。
    """
    if not settings_file.exists() or not settings_file.is_file():
        return {}

    try:
        content = settings_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("读取 settings.yaml 失败，将使用默认启动参数：%s", exc)
        return {}

    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        logger.warning("解析 settings.yaml 失败，将使用默认启动参数：%s", exc)
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def _resolve_launch_options(settings_file: Path) -> dict[str, Any]:
    """
    解析 Gradio 启动参数。

    优先级：
    1. 环境变量
    2. settings.yaml 中的 server 或 launch 配置
    3. 默认值

    支持的环境变量：
    - EBANK_HOST
    - EBANK_PORT
    - EBANK_SHARE
    - EBANK_INBROWSER
    - GRADIO_SERVER_NAME
    - GRADIO_SERVER_PORT
    """
    raw_config = _load_raw_yaml_config(settings_file)
    server_cfg = raw_config.get("server", {})
    launch_cfg = raw_config.get("launch", {})

    if not isinstance(server_cfg, dict):
        server_cfg = {}
    if not isinstance(launch_cfg, dict):
        launch_cfg = {}

    host = (
            os.getenv("EBANK_HOST")
            or os.getenv("GRADIO_SERVER_NAME")
            or server_cfg.get("host")
            or launch_cfg.get("host")
            or DEFAULT_HOST
    )

    port = _parse_int(
        os.getenv("EBANK_PORT")
        or os.getenv("GRADIO_SERVER_PORT")
        or server_cfg.get("port")
        or launch_cfg.get("port"),
        DEFAULT_PORT,
        )

    share = _parse_bool(
        os.getenv("EBANK_SHARE", server_cfg.get("share", launch_cfg.get("share", DEFAULT_SHARE))),
        DEFAULT_SHARE,
    )

    inbrowser = _parse_bool(
        os.getenv(
            "EBANK_INBROWSER",
            server_cfg.get("inbrowser", launch_cfg.get("inbrowser", DEFAULT_INBROWSER)),
        ),
        DEFAULT_INBROWSER,
    )

    return {
        "server_name": str(host).strip() or DEFAULT_HOST,
        "server_port": port,
        "share": share,
        "inbrowser": inbrowser,
    }


def _build_chat_service() -> ChatService:
    """
    构建主业务服务。
    """
    try:
        service = ChatService()
    except Exception as exc:
        raise AppStartupError("ChatService 初始化失败。") from exc

    return service


def _inject_chat_service(chat_service: ChatService) -> None:
    """
    将 main.py 中构建好的 ChatService 注入 UI 层。

    说明：
    - app/ui/app.py 内部默认通过 _get_chat_service() 获取单例服务；
    - 这里在启动入口显式注入，便于统一初始化流程，也便于后续 PyInstaller 打包。
    """
    try:
        ui_app_module._get_chat_service = lambda: chat_service  # type: ignore[assignment]
    except Exception as exc:
        raise AppStartupError("向 UI 层注入 ChatService 失败。") from exc


def main() -> int:
    """
    应用启动入口。

    启动流程：
    1. 初始化日志
    2. 加载配置
    3. 构建 ChatService
    4. 构建 Gradio Blocks 应用
    5. 按配置或环境变量启动服务
    """
    # 先用环境变量初始化日志，保证配置加载阶段的异常也能输出
    log_level = os.getenv("APP_LOG_LEVEL", "INFO")
    setup_logging(log_level)

    try:
        settings = get_settings()
    except AppConfigError as exc:
        logger.exception("加载应用配置失败：%s", exc)
        return 1
    except Exception as exc:
        logger.exception("配置加载阶段出现未预期异常：%s", exc)
        return 1

    # 若希望后续支持从 settings.yaml 扩展日志级别，可以在这里二次初始化
    logger.info("配置加载完成：app_env=%s, project_root=%s", settings.app_env, settings.paths.project_root)

    try:
        chat_service = _build_chat_service()
        _inject_chat_service(chat_service)
        demo = ui_app_module.build_demo()
    except Exception as exc:
        logger.exception("应用组件初始化失败：%s", exc)
        return 1

    launch_options = _resolve_launch_options(settings.paths.settings_file)
    logger.info(
        "准备启动 Gradio：host=%s, port=%s, share=%s, inbrowser=%s",
        launch_options["server_name"],
        launch_options["server_port"],
        launch_options["share"],
        launch_options["inbrowser"],
    )

    try:
        demo.launch(
            server_name=launch_options["server_name"],
            server_port=launch_options["server_port"],
            share=launch_options["share"],
            inbrowser=launch_options["inbrowser"],
        )
    except Exception as exc:
        logger.exception("Gradio 启动失败：%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())