"""The OAK-plus-model grounding backend.

For one label, this searches the configured ontologies with OAK
(``basic_search``) to gather candidate terms, then asks a model to choose the one
that denotes the same concept (``grounding_llm``). It implements the
:class:`~deep_research_client.claims.grounding.Grounder` protocol, so the
orchestration in ``grounding`` drives it without knowing any of this.

OAK and ``linkml-term-validator`` are an optional dependency (the ``terms``
extra); they are imported lazily, so importing this module costs nothing until a
grounder is built. OAK search is blocking and often networked, so each search
runs in a worker thread and, because OAK adapters are not assumed thread-safe,
one search runs at a time; the model choices still overlap.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from ..ontologies import DEFAULT_ONTOLOGY_ADAPTER, OntologyConfig
from .grounding import GroundingCandidate, GroundedTerm
from .grounding_llm import GROUNDING_PROMPT_VERSION, DEFAULT_GROUNDING_MAX_TOKENS, choose_candidate
from .models import GrounderInfo

logger = logging.getLogger(__name__)

__all__ = ["GroundingError", "OakGrounder"]


class GroundingError(RuntimeError):
    """Grounding could not proceed -- for example every ontology search failed."""

TERM_INSTALL_HINT = (
    "linkml-term-validator is required for ontology grounding. "
    'Install it with: pip install "deep_research_client[terms]"'
)

#: Candidates gathered per ontology before the model chooses. Enough to include
#: the right term and its near neighbours without flooding the prompt or making
#: too many lookups against a networked adapter.
DEFAULT_CANDIDATES_PER_ONTOLOGY = 3

#: The outage floor: with fewer configured ontologies than this, this many
#: searches must fail (and none succeed) before the run gives up, so a single
#: transient error on a one-ontology config does not abort everything.
_MIN_FAILS_BEFORE_OUTAGE = 3


class OakGrounder:
    """Ground labels by OAK ontology search plus a model's choice.

    Args:
        ontologies: Which ontologies to search, and how each is reached.
        llm_client: An ``openai.AsyncOpenAI``-compatible client for the choice.
        model: Model name for the choice.
        adapter: Default OAK adapter for a config entry with no value.
        cache_dir: Where OAK caches labels between runs, if the adapter caches.
        offline: Build a cache-only resolver that never reaches the network.
        candidates_per_ontology: How many search hits to keep per ontology.
        include_definitions: Fetch candidate definitions for the prompt. Off by
            default: over a networked adapter (``ols:``) it adds lookups that
            dominate runtime, and labels alone are usually enough for the model to
            choose. Worth turning on with a local ``sqlite:obo:`` adapter, where it
            is cheap.
        max_tokens: Reply budget for the choice.
    """

    def __init__(
        self,
        ontologies: OntologyConfig,
        *,
        llm_client: Any,
        model: str,
        adapter: str = DEFAULT_ONTOLOGY_ADAPTER,
        cache_dir: Optional[Path] = None,
        offline: bool = False,
        candidates_per_ontology: int = DEFAULT_CANDIDATES_PER_ONTOLOGY,
        include_definitions: bool = False,
        max_tokens: int = DEFAULT_GROUNDING_MAX_TOKENS,
    ) -> None:
        self._ontologies = ontologies
        self._llm_client = llm_client
        self._model = model
        self._adapter = adapter
        self._cache_dir = cache_dir
        self._offline = offline
        self._candidates_per_ontology = candidates_per_ontology
        self._include_definitions = include_definitions
        self._max_tokens = max_tokens
        self._access: Any = None
        self._search_lock = threading.Lock()
        self._models_seen: list[str] = []
        #: Per-ontology searches that returned, and that raised. If every search
        #: raised and none returned, the run hit an outage or a bad adapter rather
        #: than finding nothing, and the caller can tell the two apart.
        self.searches_ok = 0
        self.searches_failed = 0

    def _build_access(self) -> Any:
        """The OAK-backed resolver, built once. Raises with a hint if the extra is absent."""
        try:
            from linkml_term_validator.utils import OntologyAccess
        except ImportError as exc:  # pragma: no cover - the CLI checks find_spec before here
            raise ImportError(TERM_INSTALL_HINT) from exc
        access = OntologyAccess(
            oak_adapter_string=self._adapter,
            cache_labels=True,
            cache_dir=self._cache_dir,
            offline=self._offline,
        )
        access.oak_config = self._ontologies.resolved_adapters(self._adapter)
        return access

    def _candidates(self, label: str) -> list[GroundingCandidate]:
        """Search every configured ontology for the label.

        A prefix whose adapter is the empty string is named but not searched, as
        in term validation. A search that fails for one ontology is logged and
        skipped rather than aborting the whole grounding; the run-wide
        ``searches_ok``/``searches_failed`` counters let the caller notice when
        *every* search failed (an outage or a bad adapter) rather than nothing
        matching.

        Runs under a lock because OAK adapters are not assumed thread-safe and
        several terms are grounded at once; the lock also guards the one-time lazy
        build of the shared resolver. It is taken here, inside the worker thread,
        never across an ``await`` -- holding it across one would block the event
        loop and deadlock the concurrent calls.
        """
        with self._search_lock:
            return self._search(label)

    def _search(self, label: str) -> list[GroundingCandidate]:
        """The ontology search itself; callers hold ``_search_lock``."""
        if self._access is None:
            self._access = self._build_access()
        adapters = self._ontologies.resolved_adapters(self._adapter)
        found: dict[str, GroundingCandidate] = {}
        for prefix, adapter_string in adapters.items():
            if adapter_string == "":
                continue
            try:
                adapter = self._access.get_adapter(prefix)
                if adapter is None:
                    # No adapter for a configured prefix is a search that could not
                    # run, so it counts as failed -- otherwise a config where every
                    # prefix is unresolvable would silently ground nothing and never
                    # trip the outage check.
                    self.searches_failed += 1
                    logger.warning("No ontology adapter for %s; skipping", prefix)
                    continue
                curies = [c for c in list(adapter.basic_search(label))[: self._candidates_per_ontology]
                          if c not in found]
                # labels()/definitions() yield (curie, value[, metadata]) across OAK
                # adapters; take the first two elements so a 3-tuple adapter works too.
                labels = {t[0]: t[1] for t in adapter.labels(curies)} if curies else {}
                definitions = (
                    {t[0]: t[1] for t in adapter.definitions(curies)}
                    if curies and self._include_definitions else {}
                )
                for curie in curies:
                    found[curie] = GroundingCandidate(
                        id=curie,
                        label=labels.get(curie) or curie,
                        definition=definitions.get(curie),
                    )
                self.searches_ok += 1
            except Exception as exc:  # noqa: BLE001 - one ontology failing must not abort the run
                self.searches_failed += 1
                logger.warning("Ontology search failed for %s on %s: %s", label, prefix, exc)
        # Fail fast on a total outage: if enough searches have been attempted and not
        # one has ever succeeded, stop rather than make the same failing calls for
        # every remaining label. The floor keeps a single transient error on a
        # one-ontology config from aborting the whole run.
        searchable = sum(1 for a in adapters.values() if a != "")
        if searchable and self.searches_ok == 0 and self.searches_failed >= max(searchable, _MIN_FAILS_BEFORE_OUTAGE):
            raise GroundingError(
                "Every ontology search has failed; nothing can be grounded. Check network "
                "connectivity, the --adapter, and the ontology config (last errors are logged)."
            )
        return list(found.values())

    async def aground_term(self, label: str, context: str) -> Optional[GroundedTerm]:
        """Ground one label: OAK search for candidates, then the model's choice."""
        candidates = await asyncio.to_thread(self._candidates, label)
        if not candidates:
            return None
        term, answered = await choose_candidate(
            label, context, candidates, self._llm_client, self._model, max_tokens=self._max_tokens,
        )
        if answered and answered not in self._models_seen:
            self._models_seen.append(answered)
        return term

    def provenance(self, considered: int, grounded: int) -> GrounderInfo:
        """How this grounder ran, for the claim set's ``grounder`` slot."""
        answered = [m for m in self._models_seen if m != self._model]
        return GrounderInfo(
            name="oak-llm",
            model=", ".join(answered) or self._model,
            prompt_version=GROUNDING_PROMPT_VERSION,
            ontologies=self._ontologies.prefixes,
            considered=considered,
            grounded=grounded,
        )
