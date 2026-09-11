"""FLY//ECON — Drosophila MaleCNS v1.0 connectome makes CS MR12 economy decisions."""

from __future__ import annotations

import structlog

__version__ = "0.1.0"


def configure_logging() -> None:
    """Initialize structlog with JSON output for all module logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Configure logging on import
configure_logging()
