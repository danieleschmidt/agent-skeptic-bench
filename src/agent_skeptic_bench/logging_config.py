"""Logging configuration for Agent Skeptic Bench."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'scenario_id'):
            log_data['scenario_id'] = record.scenario_id
        if hasattr(record, 'agent_id'):
            log_data['agent_id'] = record.agent_id
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        if hasattr(record, 'evaluation_time_ms'):
            log_data['evaluation_time_ms'] = record.evaluation_time_ms
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class BenchmarkLogger:
    """Centralized logger configuration for Agent Skeptic Bench."""

    def __init__(self,
                 log_level: str = "INFO",
                 log_dir: Path | None = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_structured: bool = False):
        """Initialize logging configuration."""
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir or Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_structured = enable_structured

        # Create logs directory
        if self.enable_file:
            self.log_dir.mkdir(exist_ok=True)

        self._configure_logging()

    def _configure_logging(self):
        """Configure logging handlers and formatters."""
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)

            if self.enable_structured:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)

            root_logger.addHandler(console_handler)

        # Configure file handler
        if self.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "agent_skeptic_bench.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(self.log_level)

            if self.enable_structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)

            root_logger.addHandler(file_handler)

        # Configure error file handler
        if self.enable_file:
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "errors.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)

            if self.enable_structured:
                error_handler.setFormatter(StructuredFormatter())
            else:
                error_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s'
                )
                error_handler.setFormatter(error_formatter)

            root_logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)


class BenchmarkLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding benchmark-specific context."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any]):
        super().__init__(logger, extra)

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple:
        """Process log message with extra context."""
        # Add extra fields to the log record
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


class EvaluationLogger:
    """Specialized logger for evaluation events."""

    def __init__(self, logger_name: str = "agent_skeptic_bench.evaluation"):
        self.logger = logging.getLogger(logger_name)

    def log_evaluation_start(self, scenario_id: str, agent_id: str, session_id: str | None = None):
        """Log start of evaluation."""
        self.logger.info(
            f"Starting evaluation of scenario {scenario_id} with agent {agent_id}",
            extra={
                'scenario_id': scenario_id,
                'agent_id': agent_id,
                'session_id': session_id,
                'event_type': 'evaluation_start'
            }
        )

    def log_evaluation_complete(self, scenario_id: str, agent_id: str,
                              metrics: dict[str, float], evaluation_time_ms: int,
                              passed: bool, session_id: str | None = None):
        """Log completion of evaluation."""
        self.logger.info(
            f"Completed evaluation of scenario {scenario_id}: passed={passed}, "
            f"overall_score={metrics.get('overall_score', 0):.3f}, time={evaluation_time_ms}ms",
            extra={
                'scenario_id': scenario_id,
                'agent_id': agent_id,
                'session_id': session_id,
                'metrics': metrics,
                'evaluation_time_ms': evaluation_time_ms,
                'passed': passed,
                'event_type': 'evaluation_complete'
            }
        )

    def log_evaluation_error(self, scenario_id: str, agent_id: str, error: Exception,
                           session_id: str | None = None):
        """Log evaluation error."""
        self.logger.error(
            f"Evaluation failed for scenario {scenario_id} with agent {agent_id}: {error}",
            extra={
                'scenario_id': scenario_id,
                'agent_id': agent_id,
                'session_id': session_id,
                'error_type': type(error).__name__,
                'event_type': 'evaluation_error'
            },
            exc_info=True
        )

    def log_batch_evaluation(self, num_scenarios: int, agent_id: str,
                           session_id: str | None = None, concurrency: int = 1):
        """Log start of batch evaluation."""
        self.logger.info(
            f"Starting batch evaluation of {num_scenarios} scenarios with agent {agent_id} "
            f"(concurrency: {concurrency})",
            extra={
                'agent_id': agent_id,
                'session_id': session_id,
                'num_scenarios': num_scenarios,
                'concurrency': concurrency,
                'event_type': 'batch_evaluation_start'
            }
        )

    def log_batch_complete(self, num_scenarios: int, num_passed: int,
                         total_time_ms: int, agent_id: str,
                         session_id: str | None = None):
        """Log completion of batch evaluation."""
        pass_rate = num_passed / num_scenarios if num_scenarios > 0 else 0
        self.logger.info(
            f"Batch evaluation complete: {num_passed}/{num_scenarios} passed "
            f"({pass_rate:.1%}), total time: {total_time_ms}ms",
            extra={
                'agent_id': agent_id,
                'session_id': session_id,
                'num_scenarios': num_scenarios,
                'num_passed': num_passed,
                'pass_rate': pass_rate,
                'total_time_ms': total_time_ms,
                'event_type': 'batch_evaluation_complete'
            }
        )


# Global logger instances
_benchmark_logger = None
_evaluation_logger = None


def setup_logging(log_level: str = "INFO",
                 log_dir: Path | None = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_structured: bool = False) -> BenchmarkLogger:
    """Setup global benchmark logging."""
    global _benchmark_logger
    _benchmark_logger = BenchmarkLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_structured=enable_structured
    )
    return _benchmark_logger


def get_benchmark_logger() -> BenchmarkLogger | None:
    """Get the global benchmark logger."""
    return _benchmark_logger


def get_evaluation_logger() -> EvaluationLogger:
    """Get the evaluation logger."""
    global _evaluation_logger
    if _evaluation_logger is None:
        _evaluation_logger = EvaluationLogger()
    return _evaluation_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
