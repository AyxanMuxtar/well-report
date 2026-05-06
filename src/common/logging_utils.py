"""
Logging setup. Use:
    from src.common.logging_utils import get_logger
    log = get_logger(__name__)
    log.info("Hello")

Logs go to both stderr (with rich formatting) and a per-run file in logs/.
"""
from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime

from rich.logging import RichHandler

from src.common.config import PROJECT_ROOT

_LOGS_DIR = PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

_LOG_FILE = _LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Rich console handler
    rich_handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    rich_handler.setLevel(logging.INFO)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)

    # File handler
    file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file_handler)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
