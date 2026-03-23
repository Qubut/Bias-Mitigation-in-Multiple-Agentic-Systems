import concurrent.futures
import contextlib
import sys
import tempfile
from pathlib import Path
from typing import IO, Any, cast

import click
import requests
from loguru import logger
from returns.iterables import Fold
from returns.result import Failure, Result, Success, safe
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column

from bias_mitigation.data.errors import (
    ConfigError,
    DirectoryError,
    DomainError,
    DownloadError,
    DownloadSummary,
)
from bias_mitigation.data.models.config import AppConfig


class TempDownload(contextlib.AbstractContextManager[IO[Any]]):
    """Context manager for temporary download files with atomic replace and cleanup."""

    def __init__(self, dest_path: Path):
        self.dest_path = dest_path
        self.tmp_path: Path | None = None
        self.tmp_file: IO[Any] | None = None

    def __enter__(self):
        self.tmp_file = tempfile.NamedTemporaryFile(
            dir=self.dest_path.parent,
            prefix=self.dest_path.stem + '.',
            suffix='.part',
            delete=False,
            mode='w+b',
        )
        self.tmp_path = Path(self.tmp_file.name)
        return self.tmp_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tmp_file:
            self.tmp_file.close()
        if self.tmp_path:
            if exc_type is None:
                self.tmp_path.replace(self.dest_path)
            else:
                self.tmp_path.unlink(missing_ok=True)
        return False  # Propagate any exceptions


def should_skip_download(dest_path: Path, force: bool) -> Result[bool, DownloadError]:
    """Pattern matching: check file existence and force flag."""
    if dest_path.exists() and not force:
        logger.info(f'Already exists (use --force to redownload): {dest_path}')
        return Success(True)
    if force and dest_path.exists():
        dest_path.unlink(missing_ok=True)
        logger.info(f'Removed existing file for redownload: {dest_path}')
    return Success(False)


def download_file(url: str, dest_path: Path, force: bool = False) -> Result[None, DownloadError]:
    """
    Download a file using functional composition pattern.

    Args:
        url: URL to download from
        dest_path: Destination file path
        force: Whether to redownload existing files

    Returns:
        Result[None, DownloadError] - Success(None) on completion or Failure with error
    """
    @safe(exceptions=(requests.RequestException, IOError))
    def _perform_download() -> None:
        """Execute download with progress visualization."""
        console = Console(force_terminal=True, color_system='truecolor')
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            with (
                TempDownload(dest_path) as tmp_file,
                Progress(
                    SpinnerColumn(),
                    TextColumn(
                        '{task.description}',
                        justify='left',
                        style='bold white',
                        table_column=Column(
                            ratio=1,
                            max_width=35,
                            overflow='ellipsis',
                        ),
                    ),
                    BarColumn(
                        bar_width=80,
                        style='bar.back',
                        complete_style='cyan',
                        finished_style='cyan bold',
                        pulse_style='cyan',
                    ),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                ) as progress,
            ):
                task = progress.add_task(dest_path.name, total=total)
                for chunk in r.iter_content(chunk_size=8192):
                    size = tmp_file.write(chunk)
                    progress.update(task, advance=size)

        logger.info(f'Downloaded: {dest_path}')

    def handle_skip_result(should_skip: bool) -> Result[None, DownloadError]:
        """Pattern match on skip condition - if True, skip download."""
        if should_skip:
            return Success(None)

        # Convert safe-decorated result to our error type
        download_result = _perform_download()
        return download_result.alt(
            lambda exc: DownloadError(
                message=f'Failed to download {url}',
                cause=cast(Exception, exc),
                url=url,
            )
        )

    return should_skip_download(dest_path, force).bind(handle_skip_result)


def handle_config_error(exc: Exception) -> Result[AppConfig, DomainError]:
    """Error recovery handler for configuration loading."""
    logger.error(f'Configuration loading failed: {exc}')
    error: DomainError = ConfigError(
        message='Failed to load configuration',
        cause=exc,
        context={'config_type': type(exc).__name__},
    )
    return Failure(error)


def ensure_output_directory(base_path: Path) -> Result[Path, DomainError]:
    """Ensure output directory exists using functional pattern."""

    @safe(exceptions=(OSError, PermissionError))
    def _create_dir() -> Path:
        base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving datasets to: {base_path.resolve()}')
        return base_path

    result = _create_dir()
    return result.alt(
        lambda exc: DirectoryError(
            message='Failed to create output directory',
            cause=cast(Exception, exc),
            path=str(base_path),
        )
    )


def process_bbq_downloads(
    session: AppConfig,
    base_path: Path,
    force: bool,
) -> Result[int, DomainError]:
    """Download all BBQ category files using functional composition."""
    bbq_dir = base_path / session.bbq.dir_name
    bbq_dir.mkdir(exist_ok=True)

    results = [
        download_file(f'{session.bbq.base_url}{cat}.jsonl', bbq_dir / f'{cat}.jsonl', force)
        for cat in session.bbq.categories
    ]

    folded = cast(Result[tuple[None, ...], DomainError], Fold.collect(results, Success(())))

    match folded:
        case Success(items):
            count = len(items)
            logger.info(f'Successfully downloaded {count} BBQ categories')
            return Success(count)
        case Failure(error):
            logger.error(f'BBQ download failed: {error}')
            return Failure(error)
        case _:
            return Failure(DownloadError(message='Unknown error parsing BBQ downloads'))


def process_stereoset_downloads(
    session: AppConfig,
    base_path: Path,
    force: bool,
) -> Result[int, DomainError]:
    """Download all StereoSet files using functional composition."""
    stereoset_dir = base_path / session.stereoset.dir_name
    stereoset_dir.mkdir(exist_ok=True)

    results = [
        download_file(str(url), stereoset_dir / filename, force)
        for filename, url in session.stereoset.files.items()
    ]

    folded = cast(Result[tuple[None, ...], DomainError], Fold.collect(results, Success(())))

    match folded:
        case Success(items):
            count = len(items)
            logger.info(f'Successfully downloaded {count} StereoSet files')
            return Success(count)
        case Failure(error):
            logger.error(f'StereoSet download failed: {error}')
            return Failure(error)
        case _:
            return Failure(DownloadError(message='Unknown error parsing StereoSet downloads'))


@click.command()
@click.option(
    '--output-dir',
    '-o',
    default='datasets',
    type=click.Path(),
    help='Directory to save the datasets (default: ./datasets)',
)
@click.option(
    '--config',
    '-c',
    default='config.yaml',
    type=click.Path(exists=True),
    help='Path to the configuration YAML file (default: config.yaml)',
)
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Redownload and overwrite existing files',
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    help='Logging verbosity',
)
def cli(output_dir: str, config: str, force: bool, log_level: str) -> None:
    """Download datasets using functional composition with Result types."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    config_path = Path(config)
    base_path = Path(output_dir)

    config_result: Result[AppConfig, DomainError] = AppConfig.from_yaml(config_path).lash(
        handle_config_error
    )

    match config_result:
        case Failure(error):
            logger.error(f'Configuration failed: {error}')
            raise click.Abort(str(error))
        case Success(cfg_val):
            cfg: AppConfig = cfg_val
        case _:
            raise click.Abort('Unknown configuration result state')

    match ensure_output_directory(base_path):
        case Failure(error):
            logger.error(f'Directory setup failed: {error}')
            raise click.Abort(str(error))
        case _:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_bbq = executor.submit(process_bbq_downloads, cfg, base_path, force)
        f_ss = executor.submit(process_stereoset_downloads, cfg, base_path, force)

        bbq_result = f_bbq.result()
        ss_result = f_ss.result()

    match (bbq_result, ss_result):
        case (Success(bbq_count), Success(ss_count)):
            summary = DownloadSummary(
                message='✨ All datasets downloaded successfully!',
                bbq_count=bbq_count,
                stereoset_count=ss_count,
            )
            click.echo(f'\n🎉 {summary}')
        case (Failure(error), _):
            logger.error(f'BBQ download failed: {error}')
            raise click.Abort(str(error))
        case (_, Failure(error)):
            logger.error(f'StereoSet download failed: {error}')
            raise click.Abort(str(error))


if __name__ == '__main__':
    cli()
