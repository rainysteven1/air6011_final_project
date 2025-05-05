from typing import Any, Dict, Optional, Union
from datetime import datetime
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
import logging
import sys
import os
import yaml


class _ColorCodes:
    """ANSI 颜色代码"""

    DEBUG = "\033[36m"
    INFO = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    RESET = "\033[0m"  # 重置颜色


class _ColoredFormatter(logging.Formatter):
    """按日志级别着色的格式化器"""

    def __init__(self, fmt: str, datefmt: str = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        ori_message = super().format(record)
        color = getattr(_ColorCodes, record.levelname, _ColorCodes.RESET)
        levelname = record.levelname
        colored_level = f"{color}{levelname}{_ColorCodes.RESET}"
        return ori_message.replace(levelname, colored_level)


class CustomLogger(Logger):
    """与Lightning兼容的自定义日志记录器"""

    def __init__(
        self,
        save_dir: str,
        name: str,
        version: Optional[Union[int, str]] = None,
        default_hp_metric: bool = True,
    ):
        super().__init__()
        self._name = name
        self._version = version or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._save_dir = save_dir
        self._default_hp_metric = default_hp_metric
        self._metrics = {}

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化底层logger
        self._init_logger()

    @property
    def log_dir(self) -> str:
        """获取日志目录"""
        return self._save_dir

    @property
    def name(self) -> str:
        """获取Logger名称"""
        return self._name

    @property
    def version(self) -> str:
        """获取当前版本标识"""
        return self._version

    @staticmethod
    def _get_console_formatter(enable_color) -> logging.Formatter:
        """控制台专用格式化器（支持颜色）"""
        base_fmt = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s"
        if enable_color:
            return _ColoredFormatter(base_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            return logging.Formatter(base_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def _init_logger(self) -> None:
        """初始化底层logger"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel("INFO")
        self.logger.propagate = False

        # 清除现有handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 添加控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_console_formatter(enable_color=True))
        self.logger.addHandler(console_handler)

        log_file = os.path.join(self.log_dir, "train.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """记录超参数"""
        # 在日志中记录超参数
        params_str = "\n".join([f"  {k}: {v}" for k, v in params.items()])
        self.logger.info(f"超参数配置:\n{params_str}")

        # 将超参数保存到文件
        hp_file = os.path.join(self.log_dir, "hparams.yaml")
        with open(hp_file, "w", encoding="utf-8") as f:
            yaml.dump(
                params, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """记录指标"""
        step_str = f" (step={step})" if step is not None else ""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Metric {step_str}: {metrics_str}")

        # 记录到内部状态
        self._metrics.update(metrics)

    @rank_zero_only
    def save(self) -> None:
        """保存日志状态"""
        # 已经在log_metrics和log_hyperparams中实时保存了，这里可以不做额外操作
        pass
