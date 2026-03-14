from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", project_root: Optional[Path | str] = None) -> Path:
    """
    初始化应用全局日志配置。

    特性：
    - 输出到控制台与文件（按运行时间生成独立文件）
    - 安全清理已有处理器，防止 Windows 文件句柄泄露及日志重复
    - 抑制常用第三方网络库的冗余日志
    """
    # 1. 解析日志级别
    level_name = (level or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    # 2. 确定日志存储目录
    # 允许外部传入根目录，若未传入，则默认使用当前执行路径
    root_path = Path(project_root) if project_root else Path.cwd()
    logs_dir = root_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 3. 生成日志文件路径
    # sys.argv[0] 在交互式环境(如 Jupyter)中可能为空或非常规字符，这里做了降级处理
    program_name = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "interactive_session"
    if program_name in ("-c", ""):
        program_name = "interactive_session"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{program_name}-{timestamp}.log"

    # 4. 配置 Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 5. 安全清理旧 Handler（解决重复输出和 Windows 占用问题）
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass

    # 6. 添加 Console Handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 7. 添加 File Handler (保留自第二版)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 8. 抑制部分第三方库过于冗长的日志 (保留自第一版)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # 建议：如果是分布式训练或使用特定框架，也可在此处添加类似 logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("日志系统已初始化，日志文件路径：%s", log_path)

    return log_path