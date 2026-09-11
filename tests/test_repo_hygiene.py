"""Checks whose subject is the repository rather than any one module.

A guard that scans every file under `src/` and `tests/` reports here, not
inside a module about eval adapters: the line pytest prints is a describer
too, and a dangling citation introduced in `test_provider_fallback.py`
failing a test named for eval adapters sends a reader to the wrong file.

That is the argument `85ae9be` made one level in, when it split the
page-guard tripwire out of a test named for `scores.tsv` columns.
"""

import re
from pathlib import Path


def _cited_names(text: str) -> list[tuple[int, str]]:
    """Every backticked `test_...` in `text`, as (line number, name).

    Citations that WRAP are resolved, which a line-by-line regex cannot do.
    The tree contains one: a comment in `tests/test_eval_cli.py` splits
    `test_the_page_quotes_the_extraction_failures_note_as_the_command_prints_it`
    mid-token across two lines, because the name does not fit at this file's
    wrap width. A per-line scan sees an unclosed backtick on one line and a
    stray closing one on the next, so it reports neither a citation nor a
    problem -- "cannot see" and "saw nothing" render identically, which is
    the `EXTRACTION_FAILED` argument `mcq.py` has been making from the start.

    This branch had already learnt it once, in `06dc4e1`: a `grep` for a
    retracted claim returned zero while the claim sat in a docstring that
    wrapped between the two words. The guard that lesson produced was then
    written with the instrument the lesson was about.

    So state is carried ACROSS lines: while a backtick is open the next
    line's text is appended with no separator, which is how a wrapped
    identifier rejoins, and a leading `#` and indentation are stripped
    first. Outside a citation the separator does not matter, since only the
    text between a pair of backticks is ever read.
    """
    names: list[tuple[int, str]] = []
    pending: str | None = None
    opened_at = 0

    for line_no, raw in enumerate(text.splitlines(), 1):
        body = re.sub(r"^\s*#\s?", "", raw).rstrip("\n")
        if pending is not None:
            # The backtick is put back, because `split` reads parity from
            # the text it is given: without it the continuation begins
            # OUTSIDE a citation and the rejoined name lands at an even
            # index, where the loop below never looks. The first version
            # did exactly that -- it joined the wrapped name correctly and
            # then failed to report it, which reads the same as having no
            # wrapped citation to find.
            body = "`" + pending + body
        else:
            opened_at = line_no

        parts = body.split("`")
        # Even index = outside a citation, odd = inside. A trailing odd part
        # means the citation is still open when the line ends.
        for i in range(1, len(parts) - 1, 2):
            match = re.fullmatch(r"(test_\w+)", parts[i])
            if match:
                names.append((opened_at, match.group(1)))
        pending = parts[-1] if len(parts) % 2 == 0 else None

    return names


def test_no_comment_cites_a_test_that_does_not_exist():
    """Every `test_...` a comment names, across `src/` and `tests/`.

    A citation is the claim that rots without either text being touched, so
    this repo pins them rather than trusting them. The first pin was written
    inside the test it protected -- it asserted its own `__name__` appeared
    in `models.py` -- which catches a RENAME and not a DELETION: delete the
    test and the checker goes with it, leaving production-code documentation
    pointing at nothing and a green suite. Deletion is the likelier rot for a
    test.

    A checker placed inside the thing it checks disappears with it. This one
    is outside all of them, and it covers a second reference the first could
    not: the derived-rates test names the recipe test in its docstring, and
    that citation was unpinned.

    Backtick-delimited, because that is what a citation looks like in this
    tree and an unquoted name in prose is not distinguishable from a
    sentence. Resolved against BOTH function names and module filenames,
    since some citations name a test MODULE rather than a function.

    Not "three of them": this docstring is scanned like any other text, so
    naming the modules here would add three citations to the count the
    sentence was making. A count asserts completeness and a list does not --
    and here the count could not even be stated without changing itself.

    `docs/` is not scanned. No page there names a test today, so nothing is
    missed; it is a scope this does not cover rather than one it checked.

    No invented name appears anywhere above: the first draft of this
    docstring illustrated the backtick rule with a made-up one, and this
    function read its own example and reported it as dangling. That is the
    third instrument on this branch to be an instance of what it measures,
    and the reason its blind spots are worth stating -- it cannot see a
    citation that wraps across lines, and it treats any backticked
    `test_`-prefixed token as a claim that something exists.
    """
    root = Path(__file__).parent.parent
    tests_dir = root / "tests"

    known: set[str] = set()
    for path in sorted(tests_dir.rglob("*.py")):
        known.add(path.stem)
        text = path.read_text(encoding="utf-8")
        known.update(re.findall(r"^def (test_\w+)", text, re.M))

    # The scanner's own behaviour first: a wrapped citation must resolve.
    # The live instance is in tests/test_eval_cli.py, but a fixture pins the
    # property rather than the coincidence that the tree contains a case.
    wrapped = (
        "    # and compared in `test_the_page_quotes_the_extraction_failur\n"
        "    # es_note_as_the_command_prints_it`, because this arm cannot.\n"
    )
    assert _cited_names(wrapped) == [
        (1, "test_the_page_quotes_the_extraction_failures_note_as_the_command_prints_it")
    ], "a citation split across two comment lines is not being rejoined"
    # The delimiters are built rather than written, because a literal pair
    # around an invented name IS a citation and this function scans its own
    # file: the first version reported its own fixture as dangling. Fourth
    # time on this branch that an instrument read its own example.
    tick = chr(96)
    same_line = f"{tick}test_on_one_line{tick} and {tick}not_a_test{tick}"
    assert _cited_names(same_line) == [(1, "test_on_one_line")], (
        "a same-line citation regressed, or a non-test token was picked up"
    )

    dangling: list[str] = []
    scanned = [*tests_dir.rglob("*.py"), *(root / "src").rglob("*.py")]
    for path in sorted(scanned):
        for line_no, cited in _cited_names(path.read_text(encoding="utf-8")):
            if cited not in known:
                where = path.relative_to(root)
                dangling.append(f"{where}:{line_no} cites `{cited}`")

    assert not dangling, (
        "these name a test that no longer exists, so a reader following "
        "them finds nothing:\n  " + "\n  ".join(dangling)
    )
