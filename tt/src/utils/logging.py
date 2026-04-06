"""
Logging configuration for the MLOps platform.
"""

import logging
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
import structlog
from structlog.processors import JSONRenderer, TimeStamper, ExceptionPrettyPrinter
from structlog.stdlib import BoundLogger, LoggerFactory
from structlog.types import EventDict, Processor

from ..core.config import settings


class CustomJSONRenderer(JSONRenderer):
    """Custom JSON renderer with additional metadata."""
    
    def __call__(self, logger: logging.Logger, name: str, event_dict: EventDict) -> str:
        # Add custom metadata
        event_dict["service"] = "mlops-platform"
        event_dict["environment"] = settings.environment.value
        event_dict["log_level"] = name.upper()
        
        # Remove structlog-specific keys
        event_dict.pop("_record", None)
        event_dict.pop("_from_structlog", None)
        
        return super().__call__(logger, name, event_dict)


def add_correlation_id(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log events."""
    from .context import get_correlation_id
    
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    
    return event_dict


def setup_logging() -> None:
    """
    Setup structured logging for the application.
    
    Returns:
        None
    """
    # Clear existing handlers
    logging.basicConfig(level=logging.NOTSET)
    logging.getLogger().handlers.clear()
    
    # Configure structlog
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_correlation_id,
        TimeStamper(fmt="iso"),
    ]
    
    if settings.logging.json_format:
        processors.append(CustomJSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(),
                exception_formatter=ExceptionPrettyPrinter()
            )
        )
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure Python logging
    log_level = getattr(logging, settings.logging.level.upper())
    
    handler: logging.Handler
    if settings.logging.file_path:
        # File handler
        log_file = Path(settings.logging.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
    else:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
    
    handler.setLevel(log_level)
    
    if settings.logging.json_format:
        formatter = logging.Formatter(
            fmt='%(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt=settings.logging.format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    
    # Get root logger and add handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    
    # Set levels for specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Set levels for libraries
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log initialization
    logger = get_logger(__name__)
    logger.info(
        "Logging initialized",
        level=settings.logging.level,
        environment=settings.environment.value,
        json_format=settings.logging.json_format
    )


def get_logger(name: str) -> BoundLogger:
    """
    Get a structlog logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        structlog BoundLogger instance
    """
    return structlog.get_logger(name)


# Context management for correlation IDs
import contextvars
import uuid
from contextlib import contextmanager

_correlation_id = contextvars.ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return _correlation_id.get()


@contextmanager
def set_correlation_id(cid: Optional[str] = None):
    """
    Context manager to set correlation ID.
    
    Args:
        cid: Correlation ID (generates new if None)
        
    Yields:
        Correlation ID
    """
    if cid is None:
        cid = str(uuid.uuid4())
    
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class CorrelationIDFilter(logging.Filter):
    """Logging filter to add correlation ID."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


# Initialize logging
setup_logging()
