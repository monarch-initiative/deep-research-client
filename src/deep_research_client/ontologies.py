"""Opt-in ontology terms: which ontologies a report should cite, and how to check them.

A research run neither asks for nor checks ontology terms unless it is given an
:class:`OntologyConfig`. The config is an OAK ``oak_config.yaml``: its
``ontology_adapters`` mapping lists the ontologies by prefix, and says how each
is looked up::

    ontology_adapters:
      HP: ols:hp                  # online, through OLS
      MONDO: sqlite:obo:mondo     # a prebuilt SQLite build, downloaded once
      MYONT: simpleobo:myont.obo  # a local file
      CHEBI:                      # no value: the default adapter
      NCIT: ""                    # named in the prompt, but not checked

The keys do two jobs. They are named in a section appended to the research
query, asking for CURIEs from those ontologies only; and they scope term
validation, which reports a CURIE from any other ontology as not configured.
The values have OAK's meaning, with one addition: an entry with no value uses
the default adapter, where OAK itself would skip it.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

#: Adapter used for an entry with no value, unless the caller names another.
DEFAULT_ONTOLOGY_ADAPTER = "ols:"


class OntologyConfig(BaseModel):
    """Which ontologies a report should cite, and how each one is looked up.

    Other keys an ``oak_config.yaml`` may carry are kept but unused, so an
    existing OAK config can be passed as it is.

    >>> config = OntologyConfig(ontology_adapters={"HP": "ols:hp", "CHEBI": None, "NCIT": ""})
    >>> config.prefixes
    ['HP', 'CHEBI', 'NCIT']
    >>> config.resolved_adapters()
    {'HP': 'ols:hp', 'CHEBI': 'ols:', 'NCIT': ''}
    """

    ontology_adapters: Dict[str, Optional[str]] = Field(
        min_length=1,
        description=(
            "Ontology prefixes mapped to OAK adapter strings. An empty value uses the "
            "default adapter; an empty string names the ontology but skips checking it."
        ),
    )

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "OntologyConfig":
        """Load a config from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The config.

        Raises:
            ValueError: If the file is not a mapping with an ``ontology_adapters`` entry.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "ontology_adapters" not in data:
            raise ValueError(
                f"{path} is not an ontology config: expected an 'ontology_adapters' "
                "mapping of prefixes to OAK adapter strings"
            )
        return cls.model_validate(data)

    @property
    def prefixes(self) -> List[str]:
        """The configured ontology prefixes, in file order."""
        return list(self.ontology_adapters)

    def resolved_adapters(self, default_adapter: str = DEFAULT_ONTOLOGY_ADAPTER) -> Dict[str, str]:
        """The adapter for each prefix, with empty entries given the default.

        Args:
            default_adapter: Adapter for an entry with no value.

        Returns:
            Prefixes mapped to adapter strings; an empty string means skip.
        """
        return {
            prefix: default_adapter if adapter is None else adapter
            for prefix, adapter in self.ontology_adapters.items()
        }

    def prompt_section(self) -> str:
        """The instruction appended to the research query.

        The form puts the label in brackets right after the CURIE, which is a
        position term validation reads labels from. It is generic rather than a
        real term, which would name an ontology that may not be configured.

        >>> print(OntologyConfig(ontology_adapters={"HP": None, "CHEBI": None}).prompt_section())
        ## Ontology terms
        <BLANKLINE>
        Where a concept you discuss belongs to one of these ontologies, give its term as a CURIE followed by the term's label in brackets, in the form `PREFIX:ID (term label)`: HP, CHEBI. Put only the term's own label in the brackets, and any comment outside them. Use no other ontologies for this. Give only identifiers you have looked up, not ones recalled from memory, and leave a concept without a term rather than guess one.
        """
        return (
            "## Ontology terms\n\n"
            "Where a concept you discuss belongs to one of these ontologies, give its term as "
            "a CURIE followed by the term's label in brackets, in the form "
            f"`PREFIX:ID (term label)`: {', '.join(self.prefixes)}. Put only the term's own "
            "label in the brackets, and any comment outside them. Use no other ontologies "
            "for this. Give only identifiers you have looked up, not ones recalled from memory, "
            "and leave a concept without a term rather than guess one."
        )
