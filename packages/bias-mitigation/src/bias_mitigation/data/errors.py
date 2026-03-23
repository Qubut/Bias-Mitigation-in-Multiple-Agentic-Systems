"""Domain-specific error types for data operations using returns library pattern."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppError:
    """Base domain error for configuration and download operations."""

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
    """Error during configuration loading or validation."""


@dataclass(frozen=True)
class DownloadError(AppError):
    """Error during file download."""

    url: str | None = None

    def __str__(self) -> str:
        """Return formatted download error."""
        msg = super().__str__()
        if self.url:
            msg += f' (URL: {self.url})'
        return msg


@dataclass(frozen=True)
class DirectoryError(AppError):
    """Error with directory creation or access."""

    path: str | None = None

    def __str__(self) -> str:
        """Return formatted directory error."""
        msg = super().__str__()
        if self.path:
            msg += f' (Path: {self.path})'
        return msg


@dataclass(frozen=True)
class DownloadSummary:
    """Summary of download operations."""

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
    """Error during parsing of dataset files."""

    file_path: str | None = None

    def __str__(self) -> str:
        """Return formatted parsing error."""
        msg = super().__str__()
        if self.file_path:
            msg += f' (File: {self.file_path})'
        return msg


@dataclass(frozen=True)
class DatabaseError(AppError):
    """Error during database operations."""


DomainError = ConfigError | DownloadError | DirectoryError | ParsingError | DatabaseError
