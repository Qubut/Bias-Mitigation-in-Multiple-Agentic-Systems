"""Domain-specific error types for dataset operations."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppError:
    """Base error payload for data-domain failures.

    Attributes:
        message: Human-readable failure summary.
        cause: Optional original exception raised by an underlying library.
        context: Optional structured metadata for debugging/reporting.
    """

    message: str
    cause: Exception | None = None
    context: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Return formatted error message."""
        msg = self.message
        if self.cause:
            msg += f' (caused by: {type(self.cause).__name__}: {self.cause})'
        if self.context:
            msg += f' {self.context}'
        return msg


@dataclass(frozen=True)
class ConfigError(AppError):
    """Raised when configuration loading, parsing, or validation fails."""


@dataclass(frozen=True)
class DownloadError(AppError):
    """Raised when a dataset download request or transfer fails.

    Attributes:
        url: Optional download URL associated with the failure.
    """

    url: str | None = None

    def __str__(self) -> str:
        """Return formatted download error."""
        msg = super().__str__()
        if self.url:
            msg += f' (URL: {self.url})'
        return msg


@dataclass(frozen=True)
class DirectoryError(AppError):
    """Raised when a required directory cannot be created or accessed.

    Attributes:
        path: Optional filesystem path related to the failure.
    """

    path: str | None = None

    def __str__(self) -> str:
        """Return formatted directory error."""
        msg = super().__str__()
        if self.path:
            msg += f' (Path: {self.path})'
        return msg


@dataclass(frozen=True)
class DownloadSummary:
    """Aggregate summary for dataset download outcomes.

    Attributes:
        message: Human-readable summary header.
        bbq_count: Number of BBQ category files downloaded.
        stereoset_count: Number of StereoSet files downloaded.
    """

    message: str
    bbq_count: int = 0
    stereoset_count: int = 0

    def __str__(self) -> str:
        """Return summary as string."""
        parts = [self.message]
        if self.bbq_count > 0:
            parts.append(f'BBQ categories: {self.bbq_count}')
        if self.stereoset_count > 0:
            parts.append(f'StereoSet files: {self.stereoset_count}')
        return ' | '.join(parts)


@dataclass(frozen=True)
class ParsingError(AppError):
    """Raised when raw dataset content cannot be parsed into models.

    Attributes:
        file_path: Optional path to the file that failed to parse.
    """

    file_path: str | None = None

    def __str__(self) -> str:
        """Return formatted parsing error."""
        msg = super().__str__()
        if self.file_path:
            msg += f' (File: {self.file_path})'
        return msg


@dataclass(frozen=True)
class DatabaseError(AppError):
    """Raised when ingestion or persistence operations fail at the database layer."""


DomainError = ConfigError | DownloadError | DirectoryError | ParsingError | DatabaseError
