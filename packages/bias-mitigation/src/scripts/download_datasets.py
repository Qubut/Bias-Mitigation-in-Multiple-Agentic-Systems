import contextlib
import tempfile
from pathlib import Path
from typing import IO, Any

import click
import requests
import yaml
from loguru import logger
from returns.result import Failure, Result, Success, safe
from rich.console import Console
from rich.progress import (
    BarColumn,
    Column,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from bias_mitigation.data.models.config import AppConfig


class TempDownload(contextlib.AbstractContextManager):
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


@safe
def load_config(config_path: Path) -> AppConfig:
    """Load and validate config."""
    with config_path.open(encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


def download_file(url: str, dest_path: Path, force: bool = False) -> Result[None, Exception]:
    """Download a file."""
    if dest_path.exists() and not force:
        logger.info(f'Already exists (use --force to redownload): {dest_path}')
        return Success(None)
    if force and dest_path.exists():
        dest_path.unlink(missing_ok=True)
        logger.info(f'Removed existing file for redownload: {dest_path}')

    @safe
    def _download() -> None:
        console = Console(force_terminal=True, color_system='truecolor')  # Force color in non-TTY
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
                    transient=False,  # Persist after finish
                ) as progress,
            ):
                task = progress.add_task(dest_path.name, total=total)
                for chunk in r.iter_content(chunk_size=8192):
                    size = tmp_file.write(chunk)
                    progress.update(task, advance=size)

        logger.info(f'Downloaded: {dest_path}')

    result = _download()
    if isinstance(result, Failure):
        logger.exception(f'Failed to download {url} → {dest_path}: {result}')
    return result


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
def cli(output_dir: Path, config: str, force: bool, log_level: str) -> None:
    """Download the full BBQ and StereoSet datasets with progress bars."""
    config_path = Path(config)
    config_result = load_config(config_path)
    if isinstance(config_result, Failure):
        e = config_result.failure()
        if isinstance(e, FileNotFoundError):
            raise click.UsageError(f'Config file not found: {config}')
        if isinstance(e, yaml.YAMLError):
            raise click.UsageError(f'Invalid YAML in config file: {e}')
        raise click.UsageError(f'Config validation failed: {e!s}')

    app_config = config_result.unwrap()  # Safe after check

    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving datasets to: {base_path.resolve()}')

    bbq_dir = base_path / 'bbq'
    bbq_dir.mkdir(exist_ok=True)
    for cat in app_config.bbq.categories:
        url = f'{app_config.bbq.base_url}{cat}.jsonl'
        dest = bbq_dir / f'{cat}.jsonl'
        result: Result[None, Exception] = download_file(url, dest, force)
        if isinstance(result, Failure):
            e = result.failure()
            raise click.Abort(f'Download failed: {e}') from e

    stereoset_dir = base_path / 'stereoset'
    stereoset_dir.mkdir(exist_ok=True)
    for filename, url in app_config.stereoset.files.items():
        dest = stereoset_dir / filename
        result = download_file(str(url), dest, force)
        if isinstance(result, Failure):
            e = result.failure()
            raise click.Abort(f'Download failed: {e}') from e

    click.echo('\n🎉 All datasets downloaded successfully!')


if __name__ == '__main__':
    cli()
