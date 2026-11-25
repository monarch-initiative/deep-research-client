"""Unsloth Studio provider: deep research on a local, open model.

`Unsloth Studio <https://unsloth.ai>`_ is a desktop app that serves local
models (GGUF through llama.cpp, among others). Besides an OpenAI-compatible
``/v1`` endpoint it has a Deep Research feature: the loaded model writes a
plan, then searches the web and reads pages step by step, and writes a report
whose inline links point only at sources it collected. This provider drives
that feature over Studio's HTTP API, so the retrieval is Studio's, the model
is whatever Studio has loaded, and nothing leaves the machine except the web
searches and page fetches themselves.

The research-run routes are the ones Studio's own interface uses. They are not
a documented public API, so this module follows what Studio 2026.11.11 serves
(read from its ``/openapi.json`` and exercised against a running instance):

1. ``POST /api/chat/threads`` and ``PUT .../messages/{id}`` store the question
   as a chat message, because a run is always attached to one.
2. ``POST /api/chat/research-runs`` starts the run; the model writes a plan.
3. The run waits in ``awaiting_approval``. A person approves the plan in the
   interface; here ``POST .../approve`` does, since nobody is watching.
4. ``GET /api/chat/research-runs/{id}`` is polled until the run is
   ``completed``, ``failed`` or ``cancelled``.

Connections to Studio sometimes time out, for a minute or more at a stretch
while a run is working and now and then when it is idle (seen from WSL to
Studio on Windows). A connection that times out is therefore retried rather
than treated as a failure (see :data:`UNSLOTH_BUSY_GRACE`).

Deleting a run's chat thread deletes the run with it, so a result carries the
run's ID only when ``keep_thread`` leaves something for the ID to point at.

To use Studio only as the model behind this package's own agent, point the
ToolUniverse provider at its ``/v1`` endpoint instead; see
docs/reference/providers.md.
"""

import asyncio
from datetime import datetime
import logging
import re
import time
from typing import Any, Optional
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from . import ResearchProvider
from ..exceptions import (
    MAX_DETAIL_CHARS,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    ProviderTransientError,
    classify_status,
    truncate_detail,
)
from ..model_cards import ProviderModelCards, create_unsloth_model_cards
from ..models import ProviderConfig, ProviderHealth, ResearchResult
from ..provider_params import UnslothParams

logger = logging.getLogger(__name__)

__all__ = [
    "StudioRun",
    "UNSLOTH_DEFAULT_BASE_URL",
    "UnslothProvider",
    "cited_sources",
    "link_bare_citations",
    "loaded_model",
    "run_result",
    "studio_root",
]

#: Where Studio's desktop app listens unless told otherwise.
UNSLOTH_DEFAULT_BASE_URL = "http://127.0.0.1:8888"
#: Default deadline for a whole run when ``ProviderConfig.timeout`` is unset,
#: named once so the deadline and ``timeout_meaning`` cannot drift.
UNSLOTH_DEFAULT_TIMEOUT = 1800
#: Seconds allowed for one HTTP request. Every call here is a short read or
#: write of run state; the research itself happens inside Studio.
UNSLOTH_REQUEST_TIMEOUT = 30
#: Seconds to wait for a connection before trying a fresh one.
UNSLOTH_CONNECT_TIMEOUT = 10
#: How long connections to Studio may keep timing out before the provider
#: gives up on it. Stretches of over a minute were observed during runs.
UNSLOTH_BUSY_GRACE = 300
#: How long the first call of a run waits for an answer. Before anything has
#: been started, silence more likely means Studio is not running than that it
#: is busy; on WSL with mirrored networking a closed port times out instead
#: of refusing. Long enough to ride out one stray connection timeout.
UNSLOTH_FIRST_CONTACT_GRACE = 20

#: Failures that leave it unknown whether Studio received the request. A read
#: can simply be asked again; a write cannot, since it may have been applied.
_UNANSWERED_REQUEST = (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
#: A bracketed label with no link target after it: ``[Some title]``.
_UNLINKED_LABEL = re.compile(r"\[([^\[\]]+)\](?!\()")
_PMC_ID = re.compile(r"PMC\d+")
_TYPOGRAPHIC_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


class StudioSource(BaseModel):
    """A web page a research run collected."""

    model_config = ConfigDict(extra="ignore")

    url: str
    title: str = ""


class StudioStep(BaseModel):
    """One search or page fetch a research run performed."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    status: str = ""
    result: Optional[dict[str, Any]] = None


class StudioRun(BaseModel):
    """A research run as ``GET /api/chat/research-runs/{id}`` returns it.

    Only the fields this provider reads are declared; Studio returns more, and
    may add to them.

    >>> run = StudioRun.model_validate({"id": "r1", "status": "awaiting_approval",
    ...                                 "planRevision": 1, "planHash": "ab"})
    >>> (run.status, run.plan_revision, run.report, run.sources)
    ('awaiting_approval', 1, None, [])
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    status: str
    plan: Optional[dict[str, Any]] = None
    plan_revision: Optional[int] = Field(default=None, alias="planRevision")
    plan_hash: Optional[str] = Field(default=None, alias="planHash")
    error: Optional[Any] = None
    report: Optional[str] = None
    steps: list[StudioStep] = Field(default_factory=list)
    sources: list[StudioSource] = Field(default_factory=list)


def _may_retry(error: httpx.TransportError, method: str) -> bool:
    """Say whether a request that failed in transit can safely be sent again.

    >>> _may_retry(httpx.ConnectTimeout("timed out"), "POST")
    True
    >>> _may_retry(httpx.ReadTimeout("timed out"), "GET")
    True
    >>> _may_retry(httpx.ReadTimeout("timed out"), "POST")
    False
    >>> _may_retry(httpx.WriteError("broken pipe"), "GET")
    False
    """
    if isinstance(error, httpx.ConnectTimeout):
        return True
    return method == "GET" and isinstance(error, _UNANSWERED_REQUEST)


def studio_root(base_url: Optional[str]) -> str:
    """Return Studio's root URL, accepting the OpenAI-style ``/v1`` form too.

    The research routes live under ``/api`` and the model list under ``/v1``,
    so the root is what is configured; a URL copied from an OpenAI-compatible
    client's settings ends in ``/v1`` and means the same server.

    >>> studio_root(None)
    'http://127.0.0.1:8888'
    >>> studio_root("http://localhost:8888/v1/")
    'http://localhost:8888'
    >>> studio_root("http://studio.lan:9000")
    'http://studio.lan:9000'
    """
    return (base_url or UNSLOTH_DEFAULT_BASE_URL).rstrip("/").removesuffix("/v1")


def _loaded_chat_models(models: list[dict[str, Any]]) -> list[str]:
    """Return the IDs of loaded chat models; image and other task models are passed over."""
    return [m["id"] for m in models if m.get("loaded") and not m.get("task")]


def loaded_model(models: list[dict[str, Any]], requested: Optional[str], provider: str = "unsloth") -> str:
    """Pick the model a run will use from Studio's ``/v1/models`` list.

    Studio serves one loaded chat model at a time, and a research run needs it
    loaded before it starts. Loading one is slow and takes the GPU, so it is
    left to the person at the machine: this names the loaded model, or says
    what to load, and never loads anything itself. Image and other non-chat
    models carry a ``task`` and are passed over.

    Args:
        models: The ``data`` list of a ``/v1/models`` response.
        requested: The ``llm`` parameter, or None to take whatever is loaded.
        provider: Provider name for the error.

    Returns:
        The ID of the loaded chat model.

    Raises:
        ProviderNotConfiguredError: If nothing is loaded, or the requested
            model is not the loaded one.

    >>> models = [{"id": "unsloth/gemma", "loaded": True},
    ...           {"id": "unsloth/phi-4", "loaded": False},
    ...           {"id": "unsloth/z-image", "loaded": True, "task": "text-to-image"}]
    >>> loaded_model(models, None)
    'unsloth/gemma'
    >>> loaded_model(models, "unsloth/gemma")
    'unsloth/gemma'
    >>> def refusal(models, requested):
    ...     try:
    ...         loaded_model(models, requested)
    ...     except ProviderNotConfiguredError as error:
    ...         return error.detail
    >>> refusal(models, "unsloth/phi-4")
    "model 'unsloth/phi-4' is not loaded in Unsloth Studio (loaded: unsloth/gemma); load it there, or drop the llm parameter"
    >>> refusal([{"id": "unsloth/phi-4", "loaded": False}], None)
    'no chat model is loaded in Unsloth Studio; load one there first'
    """
    loaded = _loaded_chat_models(models)
    if not loaded:
        raise ProviderNotConfiguredError(
            provider, "no chat model is loaded in Unsloth Studio; load one there first"
        )
    if requested is None:
        return loaded[0]
    if requested not in loaded:
        raise ProviderNotConfiguredError(
            provider,
            f"model '{requested}' is not loaded in Unsloth Studio "
            f"(loaded: {', '.join(loaded)}); load it there, or drop the llm parameter",
        )
    return requested


def _title_key(title: str) -> str:
    """Fold case, spacing and typographic quotes, which models change when copying a title."""
    return " ".join(title.translate(_TYPOGRAPHIC_QUOTES).casefold().split())


def link_bare_citations(report: str, sources: list[StudioSource]) -> str:
    """Link citations the model wrote as a bare ``[label]`` naming a collected source.

    Studio rewrites the model's citations into links to collected sources, but
    leaves one as it stands when it does not recognise it, and small models
    often cite by a page's title or its PubMed Central ID instead. Two kinds
    of label are unambiguous and get the source's URL:

    - one equal to a collected source's title, apart from case, spacing and
      quote style;
    - one containing PMC IDs that all belong to a single collected source.

    Anything else is left alone: a guess would attach a statement to a page
    that may not support it.

    >>> sources = [StudioSource(url="https://a.example/1", title="LRRK2 in Parkinson's Disease - PMC"),
    ...            StudioSource(url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13568663/", title="GBA1 variants"),
    ...            StudioSource(url="https://pmc.ncbi.nlm.nih.gov/articles/PMC222/", title="Other")]
    >>> link_bare_citations("Kinase [LRRK2 in Parkinson\u2019s disease - PMC]. See [1] and [A](https://x.example).", sources)
    'Kinase [LRRK2 in Parkinson\u2019s disease - PMC](https://a.example/1). See [1] and [A](https://x.example).'
    >>> link_bare_citations("Risk rises [PMC13568663, NRR-D-25-01082].", sources)
    'Risk rises [PMC13568663, NRR-D-25-01082](https://pmc.ncbi.nlm.nih.gov/articles/PMC13568663/).'

    A label naming two collected sources, or one that was not collected, stays as written:

    >>> link_bare_citations("Both [PMC13568663, PMC222] and [PMC999].", sources)
    'Both [PMC13568663, PMC222] and [PMC999].'
    """
    by_title = {_title_key(source.title): source.url for source in sources if source.title}
    by_pmc_id = {pmc_id: source.url for source in sources for pmc_id in _PMC_ID.findall(source.url)}

    def linked(match: re.Match[str]) -> str:
        label = match.group(1)
        url = by_title.get(_title_key(label))
        if url is None:
            named = {by_pmc_id.get(pmc_id) for pmc_id in _PMC_ID.findall(label)}
            url = named.pop() if len(named) == 1 else None
        return f"{match.group(0)}({url})" if url else match.group(0)

    return _UNLINKED_LABEL.sub(linked, report)


def cited_sources(report: str, sources: list[StudioSource]) -> list[StudioSource]:
    """Return the collected sources the report links to, in order of first use.

    A run collects more pages than it cites: search hits it never opened are
    in ``sources`` too. Only a page the report links to backs a statement in
    it, so only those are citations.

    >>> sources = [StudioSource(url="https://a.example/1", title="A"),
    ...            StudioSource(url="https://b.example/2", title="B"),
    ...            StudioSource(url="https://c.example/3", title="C")]
    >>> report = "X [B](https://b.example/2). Y [A](https://a.example/1) and [B](https://b.example/2)."
    >>> [s.title for s in cited_sources(report, sources)]
    ['B', 'A']

    A URL is looked for whole, so one containing parentheses is still found:

    >>> wiki = StudioSource(url="https://en.wikipedia.org/wiki/LRRK2_(gene)", title="W")
    >>> [s.title for s in cited_sources("[W](https://en.wikipedia.org/wiki/LRRK2_(gene)).", [wiki])]
    ['W']
    """
    first_use = {source.url: report.find(f"]({source.url})") for source in sources}
    cited = {url: position for url, position in first_use.items() if position >= 0}
    by_url = {source.url: source for source in sources}
    return [by_url[url] for url in sorted(cited, key=cited.__getitem__)]


def run_result(
    run: StudioRun,
    query: str,
    *,
    provider: str,
    model: str,
    llm: str,
    instance_url: str,
    start_time: datetime,
    end_time: datetime,
    run_kept: bool = False,
) -> ResearchResult:
    """Turn a finished research run into a research result.

    Args:
        run: The run in a terminal state.
        query: The question that was asked.
        provider: Provider name to record.
        model: The model card name to record.
        llm: The Studio model that did the work.
        instance_url: Studio's root URL.
        start_time: When the provider started.
        end_time: When the run finished.
        run_kept: Whether the run still exists in Studio. Its ID is recorded
            as ``provider_job_id`` only then; the ID of a deleted run would
            be a handle on nothing.

    Returns:
        The report, with the sources it links to as citations.

    Raises:
        ValueError: If the run failed, was cancelled, or wrote no report.

    >>> now = datetime(2026, 11, 24)
    >>> def result(run, run_kept=False):
    ...     return run_result(StudioRun.model_validate(run), "q", provider="unsloth", model="m",
    ...                       llm="unsloth/gemma", instance_url="http://127.0.0.1:8888",
    ...                       start_time=now, end_time=now, run_kept=run_kept)
    >>> done = result({"id": "r1", "status": "completed", "report": "See [A](https://a.example).",
    ...                "sources": [{"url": "https://a.example", "title": "A"},
    ...                            {"url": "https://b.example", "title": "B"}]})
    >>> (done.citations, done.provider_job_id, done.run_metadata["sources_collected"])
    (['A - https://a.example'], None, 2)
    >>> result({"id": "r1", "status": "completed", "report": "Text."}, run_kept=True).provider_job_id
    'r1'
    >>> result({"id": "r1", "status": "failed", "error": "model stopped responding"})
    Traceback (most recent call last):
    ...
    ValueError: Unsloth Studio research run r1 failed: model stopped responding
    >>> result({"id": "r1", "status": "completed", "report": " "})
    Traceback (most recent call last):
    ...
    ValueError: Unsloth Studio research run r1 completed without a report
    """
    if run.status != "completed":
        reason = f": {run.error}" if run.error else ""
        raise ValueError(f"Unsloth Studio research run {run.id} {run.status}{reason}")
    if not run.report or not run.report.strip():
        raise ValueError(f"Unsloth Studio research run {run.id} completed without a report")
    report = link_bare_citations(run.report, run.sources)
    cited = cited_sources(report, run.sources)
    return ResearchResult(
        markdown=report,
        citations=[f"{s.title} - {s.url}" if s.title else s.url for s in cited],
        provider=provider,
        query=query,
        title=(run.plan or {}).get("title"),
        model=model,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
        provider_job_id=run.id if run_kept else None,
        provider_instance_url=instance_url,
        run_metadata={
            "llm": llm,
            "steps": [
                {"action": step.result.get("action"), "input": step.result.get("input")}
                for step in run.steps
                if step.result
            ],
            "sources_collected": len(run.sources),
            "sources_cited": len(cited),
        },
    )


class UnslothProvider(ResearchProvider):
    """Run Unsloth Studio's Deep Research on the model it has loaded.

    ``ProviderConfig.api_key`` is a Studio API key (``sk-unsloth-...``, made in
    Studio's settings) and ``base_url`` is where Studio listens. Auto-detection
    reads them from ``UNSLOTH_API_KEY`` and ``UNSLOTH_BASE_URL``.

    >>> provider = UnslothProvider(ProviderConfig(name="unsloth", api_key="sk-unsloth-x"))
    >>> (provider.is_available(), provider.base_url, provider.model)
    (True, 'http://127.0.0.1:8888', 'unsloth-studio-research')
    >>> UnslothProvider(ProviderConfig(name="unsloth")).unavailable_reason()
    'no Unsloth Studio API key configured (set UNSLOTH_API_KEY)'
    """

    credential_env_var = "UNSLOTH_API_KEY"
    credential_label = "Unsloth Studio (local models)"
    credential_noun = "Unsloth Studio"
    #: What a run produces depends on whichever local model is loaded, often a
    #: small one. A hosted provider that failed must not be quietly replaced
    #: by that; naming this provider as a fallback still works.
    automatic_fallback = False
    timeout_meaning = (
        "how long the whole research run may take before it is cancelled in "
        f"Studio (default {UNSLOTH_DEFAULT_TIMEOUT} s)"
    )

    def __init__(self, config: ProviderConfig, params: Optional[UnslothParams] = None):
        """Initialize the provider without contacting Studio.

        Args:
            config: Provider configuration; ``api_key`` is a Studio API key
            params: Optional Unsloth parameters
        """
        self.params = params or UnslothParams()
        super().__init__(config, self.params.model)

    @property
    def base_url(self) -> str:
        """Studio's root URL, without a trailing slash or ``/v1``."""
        return studio_root(self.config.base_url)

    def get_default_model(self) -> str:
        """Return the research model card identity, distinct from the LLM ID."""
        return "unsloth-studio-research"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Return the model card describing Studio's Deep Research."""
        return create_unsloth_model_cards()

    def _client(self) -> httpx.AsyncClient:
        """Build an HTTP client authenticated with the configured key."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=httpx.Timeout(UNSLOTH_REQUEST_TIMEOUT, connect=UNSLOTH_CONNECT_TIMEOUT),
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Read Studio's error body, in either of the two shapes it uses.

        >>> UnslothProvider._error_detail(httpx.Response(401, json={"detail": "Not authenticated"}))
        'Not authenticated'
        >>> UnslothProvider._error_detail(httpx.Response(401, json={"error": {"message": "Bad key"}}))
        'Bad key'
        >>> UnslothProvider._error_detail(httpx.Response(502, text="Bad Gateway"))
        'Bad Gateway'
        """
        is_json = response.headers.get("content-type", "").startswith("application/json")
        body = response.json() if is_json else None
        if isinstance(body, dict):
            if isinstance(body.get("detail"), str):
                return body["detail"]
            if isinstance(body.get("error"), dict) and isinstance(body["error"].get("message"), str):
                return body["error"]["message"]
        return truncate_detail(response.text, MAX_DETAIL_CHARS)

    async def _call(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        body: Optional[dict[str, Any]] = None,
        *,
        grace: float = UNSLOTH_BUSY_GRACE,
    ) -> Any:
        """Make one API call, waiting out a Studio that is not answering.

        A connection that timed out is retried for any request: it was never
        sent, so repeating it is safe for writes as well as reads. A read that
        got no answer is retried too, so one slow poll does not end a healthy
        run. Any other transport failure is raised as a provider error rather
        than a raw ``httpx`` exception; a refused connection means Studio is
        not running and fails at once.

        Args:
            client: The authenticated HTTP client
            method: HTTP method
            path: Path under Studio's root
            body: JSON body, if any
            grace: Seconds to go on retrying a Studio that does not answer

        Returns:
            The decoded JSON response

        Raises:
            ProviderNotConfiguredError: If the connection is refused
            ProviderTransientError: If Studio does not answer for ``grace``
                seconds, or the request fails in transit
            ProviderError: A classified subtype for 401/402/403/429/5xx
            ValueError: For any other failure status
        """
        give_up_at = time.monotonic() + grace
        while True:
            try:
                response = await client.request(method, path, json=body)
            except httpx.ConnectError as error:
                raise ProviderNotConfiguredError(
                    self.name, f"cannot connect to Unsloth Studio at {self.base_url}; is it running?",
                ) from error
            except httpx.TransportError as error:
                if not _may_retry(error, method):
                    raise ProviderTransientError(
                        self.name, f"{method} {path} failed in transit ({type(error).__name__})",
                    ) from error
                if time.monotonic() >= give_up_at:
                    raise ProviderTransientError(
                        self.name,
                        f"Unsloth Studio at {self.base_url} did not answer "
                        f"for {grace} s ({type(error).__name__}); is it running?",
                    ) from error
                logger.debug(f"Unsloth Studio did not answer ({type(error).__name__}); retrying {method} {path}")
                continue
            break
        if not response.is_success:
            detail = self._error_detail(response)
            classified = classify_status(self.name, response.status_code, detail)
            if classified is not None:
                raise classified
            raise ValueError(f"Unsloth Studio API error: {response.status_code} - {detail}")
        return response.json()

    async def _resolve_llm(self, client: httpx.AsyncClient) -> str:
        """Return the Studio model the run will use, or raise naming what to load.

        This is a run's first call, so it gives up on a silent Studio quickly.
        """
        models = await self._call(client, "GET", "/v1/models", grace=UNSLOTH_FIRST_CONTACT_GRACE)
        return loaded_model(models["data"], self.params.llm, self.name)

    def _run_request(self, thread_id: str, message_id: str, llm: str) -> dict[str, Any]:
        """Build the body that starts a research run.

        >>> provider = UnslothProvider(
        ...     ProviderConfig(name="unsloth", api_key="k"),
        ...     UnslothParams(max_steps=6, allowed_domains=["nih.gov"], temperature=0.1),
        ... )
        >>> request = provider._run_request("t1", "m1", "unsloth/gemma")
        >>> request["inferenceRequest"]
        {'model': 'unsloth/gemma', 'temperature': 0.1}
        >>> (request["budgets"], request["websitePolicy"])
        ({'maxSteps': 6, 'maxSources': 40}, {'allowedDomains': ['nih.gov'], 'blockedDomains': []})
        >>> "instructions" in request
        False
        """
        inference: dict[str, Any] = {"model": llm}
        if self.params.temperature is not None:
            inference["temperature"] = self.params.temperature
        request: dict[str, Any] = {
            "threadId": thread_id,
            "userMessageId": message_id,
            "inferenceRequest": inference,
            "budgets": {"maxSteps": self.params.max_steps, "maxSources": self.params.max_sources},
            "websitePolicy": {
                "allowedDomains": self.params.allowed_domains,
                "blockedDomains": self.params.blocked_domains,
            },
        }
        if self.params.system_prompt:
            request["instructions"] = self.params.system_prompt
        return request

    async def _start_run(self, client: httpx.AsyncClient, query: str, llm: str, thread_id: str) -> str:
        """Store the question as a chat message in a new thread and start a run on it.

        Returns:
            The run's ID
        """
        message_id = f"drc-{uuid4().hex}"
        created_at = int(time.time() * 1000)
        await self._call(client, "POST", "/api/chat/threads", {
            "id": thread_id,
            "title": f"deep-research-client: {query[:60]}",
            "modelType": "base",
            "modelId": llm,
            "createdAt": created_at,
        })
        await self._call(client, "PUT", f"/api/chat/threads/{thread_id}/messages/{message_id}", {
            "id": message_id,
            "threadId": thread_id,
            "role": "user",
            "content": [{"type": "text", "text": query}],
            "createdAt": created_at,
        })
        run = await self._call(
            client, "POST", "/api/chat/research-runs", self._run_request(thread_id, message_id, llm),
        )
        return run["id"]

    async def _wait_for_run(self, client: httpx.AsyncClient, run_id: str) -> StudioRun:
        """Approve the plan when it is ready, then poll until the run ends."""
        path = f"/api/chat/research-runs/{run_id}"
        while True:
            run = StudioRun.model_validate(await self._call(client, "GET", path))
            if run.status in _TERMINAL_STATUSES:
                return run
            if run.status == "awaiting_approval":
                logger.info(f"Approving Unsloth Studio research plan: {(run.plan or {}).get('title')}")
                await self._call(client, "POST", f"{path}/approve", {
                    "planRevision": run.plan_revision, "planHash": run.plan_hash,
                })
            await asyncio.sleep(self.params.poll_interval)

    async def _clean_up(self, client: httpx.AsyncClient, thread_id: str, unfinished_run_id: Optional[str]) -> None:
        """Cancel a run that did not finish and delete its thread, unless it is kept.

        A failure here is logged and not raised. Cleanup runs after the
        outcome is already decided, and raising would replace that outcome: a
        finished report would be lost, or the error that ended the run hidden
        behind one about tidying up.

        Args:
            client: The authenticated HTTP client
            thread_id: The chat thread made for the run
            unfinished_run_id: The run to cancel, or None if it reached an end
        """
        try:
            if unfinished_run_id is not None:
                await self._call(client, "POST", f"/api/chat/research-runs/{unfinished_run_id}/cancel")
            if not self.params.keep_thread:
                await self._call(
                    client, "DELETE", "/api/chat/threads", {"ids": [thread_id], "delete_files": False},
                )
        except (ValueError, httpx.TransportError) as error:
            logger.warning(
                f"Could not clean up Unsloth Studio thread {thread_id}; "
                f"delete it in Studio's chat list: {error}"
            )

    async def research(self, query: str) -> ResearchResult:
        """Run Studio's Deep Research on a question and return its report.

        The run is cancelled in Studio if it does not finish, whether it ran
        out of time or the caller stopped waiting, so it does not go on using
        the GPU. The chat thread made for it is deleted afterwards unless
        ``keep_thread`` is set.

        Args:
            query: The research question

        Returns:
            The report, with the sources it links to as citations

        Raises:
            ProviderNotConfiguredError: If no key is set or no model is loaded
            ProviderTimeoutError: If the run exceeds the timeout
            ValueError: If the query is empty or the run fails
        """
        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")
        self.require_available()

        deadline = self.config.timeout or UNSLOTH_DEFAULT_TIMEOUT
        start_time = datetime.now()
        thread_id = f"drc-{uuid4().hex}"
        async with self._client() as client:
            llm = await self._resolve_llm(client)
            logger.info(f"Starting Unsloth Studio research on {llm}")
            run_id: Optional[str] = None
            run: Optional[StudioRun] = None
            try:
                run_id = await self._start_run(client, query, llm, thread_id)
                async with asyncio.timeout(deadline):
                    run = await self._wait_for_run(client, run_id)
            except TimeoutError as error:
                raise ProviderTimeoutError(
                    self.name, f"research run {run_id} did not finish within {deadline}s",
                ) from error
            finally:
                await self._clean_up(client, thread_id, run_id if run is None else None)
        return run_result(
            run, query,
            provider=self.name, model=self.model, llm=llm, instance_url=self.base_url,
            start_time=start_time, end_time=datetime.now(), run_kept=self.params.keep_thread,
        )

    async def check_health(self) -> ProviderHealth:
        """Ask Studio for its models: is it running, is the key good, is one loaded.

        Returns:
            Health record for this provider
        """
        if not self.is_available():
            return ProviderHealth(
                provider=self.name, configured=False, reachable=False,
                detail=self.unavailable_reason(),
            )
        try:
            async with self._client() as client:
                response = await client.get("/v1/models")
        except httpx.TransportError as error:
            return ProviderHealth(
                provider=self.name, configured=True, reachable=False,
                detail=f"cannot reach Unsloth Studio at {self.base_url} ({type(error).__name__}); is it running?",
            )
        if not response.is_success:
            error_text = self._error_detail(response)
            classified = classify_status(self.name, response.status_code, error_text)
            detail = classified.diagnosis if classified else f"{response.status_code} {error_text}"
            return ProviderHealth(provider=self.name, configured=True, reachable=False, detail=detail)
        loaded = _loaded_chat_models(response.json()["data"])
        if not loaded:
            return ProviderHealth(
                provider=self.name, configured=True, reachable=False,
                detail="key accepted, but no chat model is loaded in Unsloth Studio",
            )
        return ProviderHealth(
            provider=self.name, configured=True, reachable=True,
            detail=f"model loaded: {loaded[0]}",
        )
