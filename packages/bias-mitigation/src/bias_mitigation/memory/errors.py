"""Custom error hierarchy for memory tooling operations.

Every exception in this module inherits from
:class:`mem0.exceptions.MemoryError`, so callers that want to catch
"anything the memory layer threw" can use a single
``except mem0.exceptions.MemoryError`` handler and pick up both errors
raised by mem0 directly and errors raised by :class:`Mem0Tools` on top
of it.  The lightweight ``message`` / ``cause`` constructor used inside
this package is preserved; mem0's richer fields
(``error_code``/``details``/``debug_info``) are filled in with
sensible defaults so callers don't have to think about them.
"""

from typing import Any

from mem0.exceptions import MemoryError as _Mem0MemoryError


class MemoryToolError(_Mem0MemoryError):
    """Base memory-tool exception with optional causal chain.

    Subclassing mem0's structured base means ``except
    mem0.exceptions.MemoryError`` catches every memory failure — whether
    raised inside mem0 or wrapped by :class:`Mem0Tools`.  Internally we
    keep the simpler ``message``/``cause`` keyword interface; the
    ``error_code`` and ``debug_info`` slots required by mem0 are
    populated from class-level defaults and the captured cause.

    Attributes:
        cause: Original exception that triggered this error, if any.
            Preserved as an attribute (alongside mem0's ``debug_info``)
            so existing call sites can still introspect ``err.cause``.
    """

    _DEFAULT_ERROR_CODE = 'BIAS_MITIGATION_MEMORY_ERROR'

    def __init__(
        self,
        *,
        message: str,
        cause: Exception | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or self._DEFAULT_ERROR_CODE,
            details=details or {},
            debug_info={'cause': repr(cause)} if cause is not None else {},
        )
        # The mem0 base class types ``message`` as ``Any``; redeclare locally
        # so ``__str__`` (and any other reader) sees the concrete ``str``.
        self.message: str = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause is None:
            return self.message
        return f'{self.message} (cause={self.cause})'


class MemoryConfigurationError(MemoryToolError):
    """Configuration or initialization failure for memory providers."""

    _DEFAULT_ERROR_CODE = 'BIAS_MITIGATION_MEMORY_CONFIG'


class MemoryStoreError(MemoryToolError):
    """Failure while storing memory entries."""

    _DEFAULT_ERROR_CODE = 'BIAS_MITIGATION_MEMORY_STORE'


class MemorySearchError(MemoryToolError):
    """Failure while searching memory entries."""

    _DEFAULT_ERROR_CODE = 'BIAS_MITIGATION_MEMORY_SEARCH'


class MemoryContractError(MemoryToolError):
    """Unexpected provider payload violating expected schema contract."""

    _DEFAULT_ERROR_CODE = 'BIAS_MITIGATION_MEMORY_CONTRACT'
