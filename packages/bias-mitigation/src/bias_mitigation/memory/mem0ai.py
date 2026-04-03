"""Mem0-backed DSPy retrieval model used in memory interventions."""

import os
import threading
from collections.abc import Callable
from typing import Any

import dspy
from loguru import logger
from mem0 import Memory

os.environ['MEM0_TELEMETRY'] = 'False'

from pydantic import SecretStr

from bias_mitigation.data.models.memory_config import Mem0Config


def _extract_secrets(data: Any) -> Any:
    """Recursively convert ``SecretStr`` values into plain strings."""
    if isinstance(data, dict):
        return {k: _extract_secrets(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_extract_secrets(v) for v in data]
    if isinstance(data, SecretStr):
        return data.get_secret_value()
    return data


def _run_with_timeout[T](operation: str, timeout_seconds: float, fn: Callable[[], T]) -> T:
    """Run a potentially blocking function with a bounded wait time."""
    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            result['value'] = fn()
        except BaseException as exc:
            error['value'] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_target, name=f'mem0-{operation}', daemon=True)
    worker.start()

    if not done.wait(timeout_seconds):
        raise TimeoutError(
            f'Mem0 operation "{operation}" exceeded timeout of {timeout_seconds:.1f}s'
        )

    if 'value' in error:
        raise RuntimeError(f'Mem0 operation "{operation}" failed') from error['value']

    return result['value']


class Mem0RM(dspy.Retrieve):
    """DSPy Retrieval Model that fetches past interaction memories directly from Mem0 vector store."""

    def __init__(self, mem0_config: Mem0Config | dict[str, Any], k: int = 5):
        """Create a Mem0 retrieval wrapper from typed or dict configuration."""
        super().__init__(k=k)
        self.init_timeout_seconds = float(os.getenv('MEM0_INIT_TIMEOUT_SECONDS', '20'))
        self.op_timeout_seconds = float(os.getenv('MEM0_OP_TIMEOUT_SECONDS', '15'))
        config_dict = (
            _extract_secrets(mem0_config.model_dump(exclude_none=True))
            if isinstance(mem0_config, Mem0Config)
            else _extract_secrets(mem0_config)
        )
        logger.info('Initializing Mem0 client with timeout guard...')
        self.memory = _run_with_timeout(
            'from_config',
            self.init_timeout_seconds,
            lambda: Memory.from_config(config_dict),
        )

    def forward(
        self,
        query_or_queries: str | list[str],
        user_id: str | None = None,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        """Search Mem0 for relevant passages and return them as DSPy prediction."""
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        search_args = {'limit': k if k is not None else self.k} | (
            {'filters': filters} if filters else {}
        )
        if user_id:
            search_args['user_id'] = user_id

        def _extract(data) -> list[str]:
            """Normalize Mem0 search responses into a list of text passages."""
            match data:
                case {'results': list(items)} | list(items):
                    return [
                        r.get('memory', r.get('text', '')) for r in items if isinstance(r, dict)
                    ]
                case _:
                    return []

        passages = [
            passage
            for q in queries
            for passage in _extract(
                _run_with_timeout(
                    'search',
                    self.op_timeout_seconds,
                    lambda q=q: self.memory.search(query=q, **search_args),
                )
            )
        ]

        return dspy.Prediction(passages=passages)

    def bypass_inject(
        self,
        messages: str | list[dict[str, Any]],
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Persist provided messages directly into Mem0 for a user/session."""
        _run_with_timeout(
            'add',
            self.op_timeout_seconds,
            lambda: self.memory.add(messages, user_id=user_id, metadata=metadata),
        )

    def clear_user_memory(self, user_id: str) -> None:
        """Clear memory for one user with timeout protection."""
        def action() -> dict[str, Any]:
            listed = self.memory.get_all(user_id=user_id, limit=10_000)
            items = listed.get('results', []) if isinstance(listed, dict) else []
            deleted = 0

            for item in items:
                memory_id = item.get('id') if isinstance(item, dict) else None
                if not memory_id:
                    continue
                self.memory.delete(memory_id)
                deleted += 1

            return {'deleted': deleted, 'user_id': user_id}

        _run_with_timeout(
            'clear',
            self.op_timeout_seconds,
            action,
        )
