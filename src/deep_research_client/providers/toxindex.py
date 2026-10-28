"""ToxIndex provider: Sage, a chemical safety and toxicology agent (issue #98).

ToxIndex (Insilica) answers hazard questions through its conversational agent,
Sage, citing curated toxicology data, a regulatory document index and live
predictions. Its bearer-authenticated API (``https://platform.toxindex.com/v1``,
OpenAPI at ``/v1/openapi.json``) can *read* a Sage conversation but cannot
*start* one: the REST spec has no route that sends a message, and the A2A
endpoint its agent card advertises answers every JSON-RPC method with
``501 {"error": "not implemented"}`` once the key is accepted (probed
2026-10-28). So :meth:`ToxIndexProvider.research` cannot run yet, and the
provider is registered as a stub.

What does work, and is wired up here:

- :meth:`ToxIndexProvider.check_health` calls ``GET /whoami``, so the key can be
  verified ahead of the submit route landing.
- :meth:`ToxIndexProvider.retrieve_conversation` turns a finished Sage
  conversation (started in the web app) into a :class:`ResearchResult`. Sage
  cites inline as ``[tc-45957cb5]``; each tag is the ``citation_id`` a tool
  result carries, so every citation resolves to the tool call that produced it.
"""

from collections import Counter
from datetime import datetime
import json
import logging
import re
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from . import ResearchProvider
from ..exceptions import MAX_DETAIL_CHARS, classify_status, truncate_detail
from ..model_cards import ProviderModelCards, create_toxindex_model_cards
from ..models import ProviderConfig, ProviderHealth, ResearchResult
from ..provider_params import ToxIndexParams

logger = logging.getLogger(__name__)

__all__ = [
    "SageMessages",
    "TOXINDEX_API_KEY_VAR",
    "TOXINDEX_DEFAULT_BASE_URL",
    "TOXINDEX_UNAVAILABLE_MESSAGE",
    "ToxIndexProvider",
    "sage_result",
]

TOXINDEX_DEFAULT_BASE_URL = "https://platform.toxindex.com/v1"

#: Where the key comes from. Deliberately not the class's ``credential_env_var``:
#: that declares a key which makes research work, and this one cannot yet.
TOXINDEX_API_KEY_VAR = "TOXINDEX_API_KEY"

#: The only conversation backend the API knows; every agent route requires it.
SAGE_BACKEND = "sage-agent"

#: Attributes our calls in ToxIndex's logs, as its API invites clients to.
HARNESS_HEADERS = {"X-ToxIndex-Agent-Harness": "deep-research-client"}

TOXINDEX_UNAVAILABLE_MESSAGE = (
    "ToxIndex has no API route that submits a question to Sage yet: the REST "
    "API only reads conversations, and its A2A endpoint answers "
    "501 'not implemented'. Ask Sage at https://platform.toxindex.com, then "
    "fetch the answer with `deep-research-client toxindex-conversation <id>`."
)

#: An inline citation in Sage's answer: ``[tc-45957cb5]``.
_CITATION_TAG = re.compile(r"\[(tc-[0-9a-f]+)\]")

#: The ``citation_id`` a tool result opens with. Tool output is text that is
#: only sometimes JSON, so read the one field rather than parse the whole body.
_CITATION_ID = re.compile(r'^\s*\{\s*"citation_id"\s*:\s*"(tc-[0-9a-f]+)"')


class SageContentBlock(BaseModel):
    """One block of a Sage message, as the API's ``ContentBlock`` schema has it."""

    type: str
    text: Optional[str] = None
    thinking: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class SageMessage(BaseModel):
    """One message of a Sage conversation (the API's ``Message`` schema)."""

    role: Literal["user", "assistant", "toolResult"]
    content: list[SageContentBlock]
    timestamp: int = Field(description="Milliseconds since the epoch")
    toolCallId: Optional[str] = None
    toolName: Optional[str] = None
    isError: bool = False

    def texts(self) -> list[str]:
        """Return the text of this message's text blocks, in order."""
        return [block.text for block in self.content if block.type == "text" and block.text]


class SageMessages(BaseModel):
    """``GET /agent/{conversation}/messages``: the history, and whether Sage is still at it."""

    messages: list[SageMessage]
    running: bool


def _cited_tool_calls(messages: list[SageMessage]) -> dict[str, tuple[SageContentBlock, SageMessage]]:
    """Map each citation id to the tool call that produced it and its result.

    Args:
        messages: A conversation's history.

    Returns:
        Citation id -> (the assistant's toolCall block, the toolResult message)
    """
    calls = {
        block.id: block
        for message in messages
        if message.role == "assistant"
        for block in message.content
        if block.type == "toolCall" and block.id
    }
    cited = {}
    for message in messages:
        if message.role != "toolResult" or not message.texts():
            continue
        match = _CITATION_ID.match(message.texts()[0])
        if match and message.toolCallId in calls:
            cited[match.group(1)] = (calls[message.toolCallId], message)
    return cited


def _format_citation(citation_id: str, call: SageContentBlock) -> str:
    """Render one citation as the tool call behind it.

    >>> block = SageContentBlock(type="toolCall", id="toolu_1",
    ...                          name="tox_chemical_identity",
    ...                          arguments={"identifier": "carbamate"})
    >>> _format_citation("tc-45957cb5", block)
    '[tc-45957cb5] tox_chemical_identity {"identifier": "carbamate"} (tool call toolu_1)'
    """
    arguments = json.dumps(call.arguments, ensure_ascii=False)
    return f"[{citation_id}] {call.name} {arguments} (tool call {call.id})"


def sage_result(
    conversation: SageMessages,
    conversation_id: str,
    provider: str = "toxindex",
    model: str = "sage",
) -> ResearchResult:
    """Build a research result from a finished Sage conversation.

    The question is the first user message and the report is the text of the
    last assistant message. Citations are the ``[tc-...]`` tags in that report,
    in order of first use, each resolved to the tool call that produced it.

    Args:
        conversation: The parsed ``/messages`` response.
        conversation_id: The id the conversation was fetched by.
        provider: Provider name to record on the result.
        model: Model name to record on the result.

    Returns:
        The research result

    Raises:
        ValueError: If the conversation is empty, still running, or did not
            end on an answer (Sage asked a question back, for instance).

    >>> sage_result(SageMessages(messages=[], running=False), "nope")
    Traceback (most recent call last):
    ...
    ValueError: ToxIndex conversation nope has no messages; the API answers an unknown id this way too, so check the id
    """
    messages = conversation.messages
    if not messages:
        raise ValueError(
            f"ToxIndex conversation {conversation_id} has no messages; the API "
            "answers an unknown id this way too, so check the id"
        )
    if conversation.running:
        raise ValueError(
            f"ToxIndex conversation {conversation_id} is still running; fetch it again once Sage has answered"
        )
    final = messages[-1]
    if final.role != "assistant" or not final.texts():
        raise ValueError(
            f"ToxIndex conversation {conversation_id} does not end on an answer "
            f"(its last message is a {final.role} message); finish it in the ToxIndex app"
        )
    questions = [message for message in messages if message.role == "user" and message.texts()]
    if not questions:
        raise ValueError(f"ToxIndex conversation {conversation_id} has no question in it")

    markdown = "\n\n".join(final.texts())
    cited = _cited_tool_calls(messages)
    used = list(dict.fromkeys(_CITATION_TAG.findall(markdown)))
    unresolved = [citation_id for citation_id in used if citation_id not in cited]
    if unresolved:
        logger.warning(
            f"{len(unresolved)} citation(s) in ToxIndex conversation {conversation_id} "
            f"match no tool result: {', '.join(unresolved)}"
        )
    tool_calls = [
        block for message in messages if message.role == "assistant"
        for block in message.content if block.type == "toolCall"
    ]
    start_time = datetime.fromtimestamp(messages[0].timestamp / 1000)
    end_time = datetime.fromtimestamp(final.timestamp / 1000)
    return ResearchResult(
        markdown=markdown,
        citations=[
            _format_citation(citation_id, cited[citation_id][0])
            for citation_id in used if citation_id in cited
        ],
        provider=provider,
        model=model,
        query="\n\n".join(questions[0].texts()),
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
        provider_config={
            "conversation_id": conversation_id,
            "retrieval_mode": "toxindex_conversation",
        },
        run_metadata={
            "tool_calls": len(tool_calls),
            "tools_used": dict(Counter(block.name for block in tool_calls)),
            "failed_tool_calls": sum(
                1 for message in messages if message.role == "toolResult" and message.isError
            ),
            "unresolved_citations": unresolved,
        },
    )


class ToxIndexProvider(ResearchProvider):
    """Stub provider for ToxIndex Sage, with conversation retrieval and a health probe.

    :meth:`research` raises until ToxIndex ships a route that submits a
    question, and :meth:`is_available` is ``False`` so the client never selects
    it. The key in ``TOXINDEX_API_KEY`` still serves :meth:`check_health` and
    :meth:`retrieve_conversation`; ``TOXINDEX_BASE_URL`` overrides the API root.
    Once a submit route ships, declare ``credential_env_var`` and drop
    ``stub_reason``: a provider declares one or the other.

    >>> ToxIndexProvider.stub_reason
    'ToxIndex - Sage has no API route to submit a question yet (A2A returns 501)'
    >>> provider = ToxIndexProvider(ProviderConfig(name="toxindex", api_key="k"))
    >>> provider.is_available()
    False
    >>> provider.base_url
    'https://platform.toxindex.com/v1'
    """

    #: The key is real and checked, but nothing it unlocks runs research yet.
    stub_reason = "ToxIndex - Sage has no API route to submit a question yet (A2A returns 501)"

    def __init__(self, config: ProviderConfig, params: Optional[ToxIndexParams] = None):
        """Initialize the provider.

        Args:
            config: Provider configuration; ``api_key`` is a ToxIndex API key
            params: Optional ToxIndex parameters
        """
        self.params = params or ToxIndexParams()
        super().__init__(config, self.params.model)

    @property
    def base_url(self) -> str:
        """The API root, without a trailing slash."""
        return (self.config.base_url or TOXINDEX_DEFAULT_BASE_URL).rstrip("/")

    def get_default_model(self) -> str:
        """Return Sage, the only agent ToxIndex serves."""
        return "sage"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Return the model cards describing Sage."""
        return create_toxindex_model_cards()

    def is_available(self) -> bool:
        """Never available: there is no route to submit a question to."""
        return False

    def unavailable_reason(self) -> str:
        """Say why research cannot run, and what can be done instead."""
        return TOXINDEX_UNAVAILABLE_MESSAGE

    async def research(self, query: str) -> ResearchResult:
        """Always raise: ToxIndex has no route that submits a question.

        Args:
            query: The research question (ignored)

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(TOXINDEX_UNAVAILABLE_MESSAGE)

    def _client(self) -> httpx.AsyncClient:
        """Build an HTTP client authenticated with the configured key.

        Raises:
            ValueError: If no key is configured
        """
        if not self.config.api_key:
            raise ValueError(f"{TOXINDEX_API_KEY_VAR} is required to call the ToxIndex API")
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}", **HARNESS_HEADERS},
            timeout=self.config.timeout or 60,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Read the API's ``{"error": ...}`` body, or the raw text when it is not one.

        >>> ToxIndexProvider._error_detail(httpx.Response(401, json={"error": "invalid api key"}))
        'invalid api key'
        >>> ToxIndexProvider._error_detail(httpx.Response(400, text='unknown backend ""'))
        'unknown backend ""'
        """
        is_json = response.headers.get("content-type", "").startswith("application/json")
        body = response.json() if is_json else None
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return body["error"]
        return truncate_detail(response.text, MAX_DETAIL_CHARS)

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise a typed error for a failed call.

        Args:
            response: The response to check

        Raises:
            ProviderError: A classified subtype for 401/402/403/429/5xx
            ValueError: For any other failure status
        """
        if response.is_success:
            return
        detail = self._error_detail(response)
        classified = classify_status(self.name, response.status_code, detail)
        if classified is not None:
            raise classified
        raise ValueError(f"ToxIndex API error: {response.status_code} - {detail}")

    async def check_health(self) -> ProviderHealth:
        """Verify the key with ``GET /whoami``, then report that research cannot run.

        ``reachable`` stays ``False`` even when the key is accepted, because it
        means "can take work" and Sage cannot be sent any yet. The detail says
        which half failed.

        Returns:
            Health record for this provider
        """
        if not self.config.api_key:
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail=f"no ToxIndex API key configured (set {TOXINDEX_API_KEY_VAR})",
            )
        async with self._client() as client:
            response = await client.get("/whoami")
        if not response.is_success:
            # Classified for the remedy text only; the record is the result.
            error = self._error_detail(response)
            classified = classify_status(self.name, response.status_code, error)
            detail = classified.diagnosis if classified else f"{response.status_code} {error}"
            return ProviderHealth(provider=self.name, configured=True, reachable=False, detail=detail)
        plan = response.json().get("plan", "unknown")
        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=False,
            detail=f"key accepted ({plan} plan), but {self.stub_reason.split(' - ', 1)[1]}",
        )

    async def fetch_conversation(self, conversation_id: str) -> SageMessages:
        """Fetch a Sage conversation's history.

        Args:
            conversation_id: Id of a conversation, as shown in the ToxIndex app URL

        Returns:
            The parsed history
        """
        async with self._client() as client:
            response = await client.get(
                f"/agent/{conversation_id}/messages", params={"backend": SAGE_BACKEND}
            )
        self._raise_for_status(response)
        return SageMessages.model_validate(response.json())

    async def retrieve_conversation(self, conversation_id: str) -> ResearchResult:
        """Turn a finished Sage conversation into a research result.

        Args:
            conversation_id: Id of a conversation, as shown in the ToxIndex app URL

        Returns:
            The research result, citations resolved to the tool calls behind them
        """
        logger.info(f"Retrieving ToxIndex conversation {conversation_id}")
        conversation = await self.fetch_conversation(conversation_id)
        return sage_result(conversation, conversation_id, provider=self.name, model=self.model)
