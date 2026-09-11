"""Checks whose subject is the repository rather than any one module.

A guard that scans every file under `src/` and `tests/` reports here, not
inside a module about eval adapters: the line pytest prints is a describer
too, and a dangling citation introduced in `test_provider_fallback.py`
failing a test named for eval adapters sends a reader to the wrong file.

That is the argument `85ae9be` made one level in, when it split the
page-guard tripwire out of a test named for `scores.tsv` columns.
"""

import inspect
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
    identifier rejoins. Indentation is stripped from every line and a
    leading `#` after it, so a wrap inside a DOCSTRING rejoins too -- the
    first version stripped the indent only when a `#` followed it, which
    covered comments and left every docstring wrap embedding whitespace in
    the middle of the name.

    The carry is BOUNDED, and that is the load-bearing part. An odd
    backtick is not always a wrap: `mcq.py` compiles `[*`#]+` and never
    restores parity, and four other modules under `src/` have such a line.
    Carrying state on those inverts parity for every remaining line of the
    file, so a real citation lands at an even index and the loop below --
    which is the same even-index failure described above, at file scale --
    reports nothing. Measured: with one stray backtick ahead of it, a
    citation two lines later was returned as `[]`.

    So a fragment is carried only when it could be PART OF AN IDENTIFIER.
    `#]+")` is not, and is dropped where the line ends; `test_the_page_
    quotes_..._note_as_` is, and continues. Giving an instrument memory
    without a condition that ends it does not shrink its blind spot, it
    relocates it -- from two lines to everything after the first
    unbalanced delimiter, and in the same silent direction.
    """
    names: list[tuple[int, str]] = []
    pending: str | None = None
    opened_at = 0

    for line_no, raw in enumerate(text.splitlines(), 1):
        body = re.sub(r"^#\s?", "", raw.strip())
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
            # A trailing `.py` comes off first: a module can be cited as a
            # bare stem, as a `tests/...` path, or as a backticked FILENAME,
            # and the third spelling matched neither resolver -- `.` is not
            # `\w` for this regex, and the path check needs the directory.
            # The tree's one instance was this file's own module docstring,
            # so the file introducing the guard opened with a citation the
            # guard could not see.
            token = re.sub(r"\.py$", "", parts[i])
            match = re.fullmatch(r"(test_\w+)", token)
            if match:
                names.append((opened_at, match.group(1)))
        # Only an identifier-shaped fragment continues; see the docstring.
        open_fragment = parts[-1] if len(parts) % 2 == 0 else ""
        identifier_shaped = re.fullmatch(r"[\w.]+", open_fragment)
        pending = open_fragment if identifier_shaped else None

    # A fragment still open at EOF is a citation that was never closed, and
    # dropping it is the silent direction this whole function argues
    # against. Reported as a name instead, so it reaches the caller's
    # resolution: if it names a real test the missing backtick costs
    # nothing, and if it does not it is listed rather than skipped.
    if pending is not None:
        match = re.fullmatch(r"(test_\w+)", re.sub(r"\.py$", "", pending))
        if match:
            names.append((opened_at, match.group(1)))

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
    function read its own example and reported it as dangling.

    That failure has a family, and this is its one home -- three sentences
    elsewhere each called their own case "the third" or "the fourth", in
    three different vocabularies, and two of the three numbers could not be
    right. Named rather than counted, because a tally kept in several files
    is a claim with nowhere to check it:

    - a `caplog` wrapper that suppressed nothing it was meant to suppress
    - a recursion counter that spent a Python frame per level it counted
    - a `#:` ragged-line audit that read every wrapped paragraph end as a
      stub, because `lstrip("#")` turns a Sphinx marker into ":"
    - `_page_guards`, which counted its own body as a page read
    - `recipe in inspect.getsource(...)`, which matched the line DEFINING
      `recipe` and so held however the comprehension was rewritten
    - this docstring's first invented example, and the fixture name below
    - `_cited_names` itself, which rejoined a wrapped citation correctly
      and then looked for it at a parity the scan never reads

    Its blind spots are worth stating for the same reason:

    - it treats any backticked `test_`-prefixed token as a claim that
      something exists
    - `known` is built from `^def test_` at column zero, so a test defined
      inside a class would be absent and every citation of it reported --
      loudly, which is the direction to prefer
    - the PATH spelling is matched per line, so a path reference that wraps
      is invisible where a wrapped backticked name is not. No instance
      today; the asymmetry is here because both checks now sit in one place
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
    # file: the first version reported its own fixture as dangling. That
    # family is listed in `_cited_names`' docstring above -- one home, no
    # ordinal, because four sentences in this tree each numbered their own
    # case and no two agreed.
    tick = chr(96)
    same_line = f"{tick}test_on_one_line{tick} and {tick}not_a_test{tick}"
    assert _cited_names(same_line) == [(1, "test_on_one_line")], (
        "a same-line citation regressed, or a non-test token was picked up"
    )

    # A stray backtick is not a wrap. `mcq.py` compiles a character class
    # containing one and never restores parity, so carrying state on it
    # inverted every remaining line of that file and the scan reported
    # nothing -- measured as `[]` for the citation below before the carry
    # was bounded to identifier-shaped fragments.
    after_stray = (
        f'PAT = re.compile(r"[*{tick}#]+")\n'
        "X = 1\n"
        f"# see {tick}test_on_one_line{tick} for why\n"
    )
    assert _cited_names(after_stray) == [(3, "test_on_one_line")], (
        "an unbalanced backtick that is not a wrap has disabled the scan "
        "for the rest of the file"
    )

    # And a wrap inside a DOCSTRING, where this repo keeps most of its
    # prose: the indent must come off the continuation or it lands in the
    # middle of the name. Stripping it only after a `#` covered comments
    # and left every docstring wrap invisible.
    in_docstring = (
        f"    {tick}test_a_name_that_wraps_\n"
        f"    here{tick} and more prose\n"
    )
    wrapped_name = "test_a_name_that_wraps_here"
    assert _cited_names(in_docstring) == [(1, wrapped_name)], (
        "a citation wrapped across two indented docstring lines is not "
        "being rejoined"
    )

    # An unclosed citation reaches the caller rather than vanishing. Only
    # the WRAP-shaped one can: a fragment followed by prose on the same line
    # is not identifier-shaped and is correctly never carried, so the case
    # that survives to EOF is a line ending right after the name -- a wrap
    # whose continuation never came. It names nothing here, so the guard
    # below lists it; the alternative was to drop it, which renders the same
    # as a file with no citations at all.
    unclosed = f"# see {tick}test_never_closed\n"
    assert _cited_names(unclosed) == [(1, "test_never_closed")], (
        "a citation left open at end of file is being dropped instead of "
        "reported"
    )

    # The third spelling of a module citation: a backticked FILENAME. The
    # bare stem resolves through `path.stem` and the `tests/...` path
    # through the regex below; this one matched neither, and the tree's only
    # instance was this file's own module docstring.
    with_extension = f"{tick}test_on_one_line.py{tick} and prose"
    assert _cited_names(with_extension) == [(1, "test_on_one_line")], (
        "a module cited as a backticked filename is invisible to the scan"
    )

    # A path reference is a citation too, and the one in `src/` that the
    # backtick rule cannot reach: `MatrixConfig.on_scores` cites
    # tests/test_eval_matrix.py by path. The backtick requirement exists
    # because an unquoted NAME in prose is not distinguishable from a
    # sentence -- which is not true of a path, so paths are resolved as
    # well, against the files that exist.
    dangling: list[str] = []
    scanned = [*tests_dir.rglob("*.py"), *(root / "src").rglob("*.py")]
    for path in sorted(scanned):
        text = path.read_text(encoding="utf-8")
        where = path.relative_to(root)
        for line_no, cited in _cited_names(text):
            if cited not in known:
                dangling.append(f"{where}:{line_no} cites `{cited}`")
        for line_no, line in enumerate(text.splitlines(), 1):
            for ref in re.findall(r"tests/(test_\w+)\.py", line):
                if ref not in known:
                    dangling.append(f"{where}:{line_no} cites tests/{ref}.py")

    assert not dangling, (
        "these name a test that no longer exists, so a reader following "
        "them finds nothing:\n  " + "\n  ".join(dangling)
    )



def _tests_whose_body_contains(needle: str) -> set[str]:
    """Names of the tests under `tests/` whose body carries `needle`.

    Attribution resets at every top-level `def`, for the reason
    `_page_guards` records: crediting a helper's text to the test above it
    is how that function came to count its own source.
    """
    found: set[str] = set()
    for path in sorted((Path(__file__).parent).rglob("*.py")):
        current: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"def (\w+)", line)
            if match:
                name = match.group(1)
                current = name if name.startswith("test_") else None
            elif needle in line and current:
                found.add(current)
    return found


def test_mcqscore_still_names_the_test_that_runs_its_recipe():
    """The citing direction, which the dangling check cannot see.

    `test_no_comment_cites_a_test_that_does_not_exist` fails when a citation
    points at nothing. It stays green when a citation is simply DELETED --
    and `MCQScore`'s class comment says naming its test is deliberate, and
    that describing the recipe without saying what runs it "is what left
    this comment unexecuted for four commits". A tidy-up that drops the name
    while keeping the recipe the other guards pin returns the comment to the
    state its own sentence calls the failure.

    Its own test rather than an assertion over there, because it is the
    opposite defect under a name about citations that point at nothing --
    the line pytest prints is a describer too.

    The SPECIFIC name, not merely that some test is cited: the first version
    asserted non-emptiness, so swapping the citation for any other existing
    test left it green while its message's premise was false, and the
    dangling check could not catch it either because the substitute
    resolves. That is the same widening the recipe check was corrected for
    two files over -- a predicate wider than the message it prints.

    The expected name is DERIVED, not written: the recipe test is whichever
    test function executes the reconstruction expression, found by reading
    `tests/` for the one whose body carries it. So a rename of that test
    still passes (it is the same test), swapping the citation for an
    unrelated test fails, and no literal name is duplicated here to go
    stale. Importing it would have been the other way, and `tests/` is not
    an importable package.

    From outside that test, so deleting it cannot delete the checker.
    """
    from deep_research_client.evaluation.models import MCQScore

    source = inspect.getsource(MCQScore)
    recipe = "{k: v for k, v in dump.items() if k in MCQScore.model_fields}"
    assert recipe in source, (
        "MCQScore's class comment no longer carries the recipe, so there is "
        "no reconstruction expression for a cited test to be running"
    )

    runs_the_recipe = _tests_whose_body_contains(f"counts = {recipe}")
    assert len(runs_the_recipe) == 1, (
        f"expected exactly one test executing the recipe, found "
        f"{sorted(runs_the_recipe)} -- this check cannot say which one "
        f"MCQScore should name"
    )
    expected = runs_the_recipe.pop()

    cited = [name for _, name in _cited_names(source)]
    assert expected in cited, (
        f"MCQScore's class comment no longer names {expected}, the test "
        f"that executes its documented recipe; the expression is still "
        f"pinned, but a reader has nothing to run it by. Cited: {cited}"
    )
