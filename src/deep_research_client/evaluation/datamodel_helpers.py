"""Small predicates over the generated data model.

`datamodel.py` is generated from `evaluation.yaml` by `gen-pydantic`, so hand
-written helpers over it live here rather than being lost on the next
regeneration.
"""

from .datamodel import MatchStyle, SpotCheck


def is_prefix_match(spec: SpotCheck) -> bool:
    """Whether a spot check compares with `prefix` rather than `exact`.

    `match` is optional with an `ifabsent` of `exact`, so three different
    inputs mean exact: the key absent, the key present as `null`, and the key
    set to `exact`. Three sites used to compare against `MatchStyle.prefix`
    directly, each silently reading `None` as exact; this says it once.

    >>> is_prefix_match(SpotCheck(name="n", pattern="x", expected="y"))
    False
    >>> is_prefix_match(SpotCheck(name="n", pattern="x", expected="y", match=None))
    False
    >>> is_prefix_match(SpotCheck(name="n", pattern="x", expected="y", match="exact"))
    False
    >>> is_prefix_match(SpotCheck(name="n", pattern="x", expected="y", match="prefix"))
    True
    """
    return spec.match == MatchStyle.prefix.value
