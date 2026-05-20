"""Compatibility shims for mem0 internals.

The mem0 SDK ships with two behaviours that are awkward for a research
codebase: it talks to PostHog for telemetry, and its OpenAI-style
embedder unconditionally sends the ``dimensions=`` parameter, which
several OpenAI-compatible servers (matryoshka-style embedding stacks,
local sglang/vLLM gateways, ...) reject with a 400. This module hides
both quirks behind two small public helpers — :func:`disable_mem0_telemetry`
and :func:`patch_openai_embedder` — so the rest of ``Mem0Tools`` can
target a single stable surface regardless of which mem0 version is
installed in the current environment.

All patches are idempotent and version-gated via
``_EMBEDDER_PATCH_VERSION`` so they can be safely re-applied after a
mem0 upgrade without compounding monkey-patches.
"""

from __future__ import annotations

import os
from typing import Any, cast

from loguru import logger

# Disable mem0/PostHog telemetry before any mem0 import-time side effects.
os.environ.setdefault('MEM0_TELEMETRY', 'False')
os.environ.setdefault('POSTHOG_DISABLED', 'true')
os.environ.setdefault('DO_NOT_TRACK', 'true')
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY', 'true')

from mem0 import Memory
from mem0.memory import main as mem0_main
from mem0.memory import telemetry as mem0_telemetry

try:
    from mem0.embeddings.openai import OpenAIEmbedding as _OpenAIEmbedding
except Exception:
    _OpenAIEmbedding = None

OpenAIEmbedding: Any = _OpenAIEmbedding


_EMBEDDER_PATCH_VERSION = 2


def disable_mem0_telemetry() -> None:
    """Replace mem0's PostHog client and capture hooks with no-ops.

    Even with telemetry env vars set, mem0 sometimes instantiates a
    PostHog client at import time (depending on version) and emits
    capture events from inside ``Memory.search`` and friends. This
    helper rewires those hooks to be inert so reproducibility runs do
    not generate background network traffic and offline / air-gapped
    environments do not see spurious connection errors.

    The call is idempotent and silent on failure: if mem0's internal
    module layout has changed and the patch cannot be applied, a debug
    log is emitted and the function returns normally so it never blocks
    startup.
    """
    try:
        no_op_type = _build_noop_posthog_type()
        _patch_posthog(no_op_type)
        _patch_capture_hooks()
        logger.info('[mem0_compat]: Disabled Mem0 telemetry capture hooks.')
    except Exception as error:
        logger.debug(f'[mem0_compat]: Unable to patch Mem0 telemetry hooks: {error}')


def patch_openai_embedder(*, force_dimensionless: bool) -> None:
    """Make the mem0 OpenAI embedder tolerate servers that reject ``dimensions=``.

    mem0's OpenAI embedder always passes ``dimensions=`` to the
    ``embeddings.create`` call, which several OpenAI-compatible servers
    reject — typically with a matryoshka- or dimension-related 400 error.
    This patch wraps ``OpenAIEmbedding.embed`` so that:

    - when ``force_dimensionless`` is ``True`` every call skips the
      parameter outright (use this when you know the endpoint does not
      support it);
    - otherwise the first call is attempted normally, and only on a
      dimensions-related failure does the wrapper retry without the
      parameter and remember the affected embedder instance so
      subsequent calls take the dimensionless path immediately.

    The patch is version-gated by ``_EMBEDDER_PATCH_VERSION`` and is
    skipped silently if mem0's OpenAI embedder is not importable in the
    current environment.

    Args:
        force_dimensionless: When ``True``, never include the
            ``dimensions`` parameter. When ``False``, opt in lazily on
            the first matryoshka/dimension-related error.
    """
    if OpenAIEmbedding is None:
        logger.debug('[mem0_compat]: Skipping embedder patch: OpenAIEmbedding unavailable.')
        return

    if (
        getattr(OpenAIEmbedding, 'bias_mitigation_dimensions_patch_version', 0)
        >= _EMBEDDER_PATCH_VERSION
    ):
        return

    original_embed = OpenAIEmbedding.embed
    OpenAIEmbedding.embed = _build_patched_embed(original_embed, force_dimensionless)
    OpenAIEmbedding.bias_mitigation_dimensions_patch_applied = True
    OpenAIEmbedding.bias_mitigation_dimensions_patch_version = _EMBEDDER_PATCH_VERSION


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _embed_without_dimensions(instance: Any, text: str) -> list[float]:
    """Call the underlying OpenAI client without the ``dimensions`` parameter.

    Used by the embedder patch as the fallback path for servers that
    reject ``dimensions=``. Newlines in the input are replaced with
    spaces to mirror mem0's own pre-processing.

    Args:
        instance: The ``OpenAIEmbedding`` instance whose ``client`` and
            ``config.model`` should be used for the request.
        text: Text to embed.

    Returns:
        The single embedding vector returned by the server.
    """
    response = instance.client.embeddings.create(
        input=[text.replace('\n', ' ')],
        model=instance.config.model,
    )
    return cast(list[float], response.data[0].embedding)


def _build_patched_embed(original_embed: Any, force_dimensionless: bool) -> Any:
    """Build the replacement ``OpenAIEmbedding.embed`` used by the patch.

    The closure keeps a set of embedder-instance ids that have already
    proven unable to handle ``dimensions=``, so once a given embedder
    fails over to the dimensionless path it stays there for the
    remainder of the process.

    Args:
        original_embed: The unbound original ``OpenAIEmbedding.embed``
            method (used for the optimistic first call).
        force_dimensionless: When ``True``, never attempt the original
            call — always use the dimensionless path.

    Returns:
        A replacement function with the same signature as
        ``OpenAIEmbedding.embed``.
    """
    forced_ids: set[int] = set()

    def _patched(instance: Any, text: str, memory_action: str | None = None) -> list[float]:
        if force_dimensionless or id(instance) in forced_ids:
            return _embed_without_dimensions(instance, text)
        try:
            return cast(list[float], original_embed(instance, text, memory_action))
        except Exception as error:
            error_text = str(error).lower()
            if 'matryoshka' not in error_text and 'dimension' not in error_text:
                raise
            forced_ids.add(id(instance))
            logger.warning(
                '[mem0_compat]: Embedder rejected `dimensions=`; switching this instance '
                'to dimensionless requests.'
            )
            return _embed_without_dimensions(instance, text)

    return _patched


def _build_noop_posthog_type() -> type:
    """Build a drop-in replacement type for mem0's PostHog client.

    The returned class mimics the small subset of the PostHog client
    surface that mem0 calls into — ``capture`` and ``shutdown`` — but
    every method is a no-op. ``disabled = True`` is set so any mem0
    code path that checks the flag short-circuits cleanly.

    Returns:
        A class object suitable for assigning to
        ``mem0_telemetry.Posthog``.
    """

    class _NoOpPosthog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.disabled = True

        def capture(self, *args: Any, **kwargs: Any) -> None:
            return

        def shutdown(self) -> None:
            return

    return _NoOpPosthog


def _noop(*args: Any, **kwargs: Any) -> None:
    """No-op replacement used to silence mem0's telemetry capture hooks."""
    return


def _patch_posthog(no_op_type: type) -> None:
    """Swap mem0's PostHog class and any live client for the no-op type.

    Args:
        no_op_type: The replacement class produced by
            :func:`_build_noop_posthog_type`.
    """
    mem0_telemetry.Posthog = no_op_type
    client = getattr(mem0_telemetry, 'client_telemetry', None)
    if client is not None:
        client.posthog = no_op_type()


def _patch_capture_hooks() -> None:
    """Rewire every ``capture_event`` entry point in mem0 to the no-op.

    mem0 imports ``capture_event`` into several modules (and sometimes
    binds it into closures such as ``Memory.search.__globals__``), so
    suppressing telemetry reliably means patching each known site
    rather than just the canonical definition.
    """
    mem0_telemetry.capture_event = _noop
    if hasattr(mem0_telemetry, 'capture_client_event'):
        mem0_telemetry.capture_client_event = _noop
    mem0_main.capture_event = _noop
    search_globals = getattr(Memory.search, '__globals__', None)
    if isinstance(search_globals, dict) and 'capture_event' in search_globals:
        search_globals['capture_event'] = _noop
