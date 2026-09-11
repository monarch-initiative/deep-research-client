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


def _unclosed_citation(fragment: str) -> str | None:
    """The test a fragment was naming when its closing backtick never came.

    `None` when the fragment names no test, which is the common case and
    the reason the gate is here: most unbalanced backticks in this tree are
    character classes and regex literals rather than citations someone
    forgot to close. Measured by removing it -- returning the fragment
    whatever its shape reported four dangling "citations" immediately,
    among them a `pip` install line and three sentences containing the word
    `eval` in backticks.
    """
    match = re.fullmatch(r"(test_\w+)", re.sub(r"\.py$", "", fragment))
    return match.group(1) if match else None


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
    leading `#` with ALL the whitespace after it, so a wrap inside a
    DOCSTRING rejoins too -- the first version stripped the indent only
    when a `#` followed it, which covered comments and left every docstring
    wrap embedding whitespace in the middle of the name, and the second
    took at most one space after the `#`, which broke the rejoin for any
    comment continuation indented under its own marker.

    The carry is BOUNDED, and that is the load-bearing part. An odd
    backtick is not always a wrap: `mcq.py` compiles `[*`#]+` and never
    restores parity, and ten other modules under `src/` have such a line.
    Carrying state on those inverts parity for every remaining line of the
    file, so a real citation lands at an even index and the loop below --
    which is the same even-index failure described above, at file scale --
    reports nothing. Measured: with one stray backtick ahead of it, a
    citation two lines later was returned as `[]`.

    So a fragment is carried only when it could be PART OF AN IDENTIFIER.
    `#]+")` is not; `test_the_page_ / quotes_..._note_as_` is, and
    continues. (The ellipsis in that illustration is what keeps it inert: a
    real-looking name written here would BE a citation, in a function whose
    own file is one of the ones it is run over.)

    A carry that ENDS without the citation closing is reported, wherever it
    ends: at the last line of the text, or mid-file, on the first line
    whose join is not identifier-shaped. Only the first of those two was
    reported at first, and it is the rarer one by a long way -- a citation
    whose closing backtick is missing is followed by more file almost
    always, and was dropped with nothing recorded. That is the silent
    direction this whole function argues against, arrived at by closing
    half of the case and reading the half as the whole.

    Reporting means handing the name to the caller's resolution rather than
    printing here: if it names a real test the missing backtick costs
    nothing, and if it does not it is listed rather than skipped. A carry
    that DOES close normally is not reported twice -- the span it resolved
    to is what the loop already appended.

    That bound is a TRADE, not a closure, and the losing side is recorded
    here because a describer that states only the win is the shape this
    branch keeps finding. Rejecting a fragment also resets parity, so a
    GENUINE multi-line code span -- one whose first line ends in something
    that is not identifier-shaped -- has its second line scanned as though
    it began outside a citation. A real citation on that line lands at an
    even index and is dropped. `models.py:856-857` is such a span today,
    and so is the `wrapped` fixture's own SOURCE -- the fixtures built with
    `chr(96)` are not, since their delimiters never reach the file. Neither
    span carries a citation after its closing backtick, which is why the
    cost is nothing yet. Measured, and pinned by a fixture: a span followed
    by a citation on its closing line returns `[]`.

    Kept anyway, because the two errors are not the same size. A stray
    backtick is common and a wrap-shaped one is not -- counted across
    `src/`, eleven modules leave a fragment open that cannot be part of an
    identifier and three leave one that can -- and carrying on a stray
    costs every remaining line of its file where a multi-line code span
    costs exactly one. It is the direction of that gap the trade rests on,
    not the two numbers, which move whenever a module is added.

    Giving an instrument memory without a condition that ends it does not
    shrink its blind spot, it relocates it -- from two lines to everything
    after the first unbalanced delimiter, in the same silent direction.
    """
    names: list[tuple[int, str]] = []
    pending: str | None = None
    opened_at = 0

    for line_no, raw in enumerate(text.splitlines(), 1):
        body = re.sub(r"^#\s*", "", raw.strip())
        carried = pending
        if carried is not None:
            # The backtick is put back, because `split` reads parity from
            # the text it is given: without it the continuation begins
            # OUTSIDE a citation and the rejoined name lands at an even
            # index, where the loop below never looks. The first version
            # did exactly that -- it joined the wrapped name correctly and
            # then failed to report it, which reads the same as having no
            # wrapped citation to find.
            body = "`" + carried + body
        else:
            opened_at = line_no

        parts = body.split("`")
        # Even index = outside a citation, odd = inside. A trailing odd part
        # means the citation is still open when the line ends.
        carry_resolved = False
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
                # Index 1 is where a carried fragment lands, so a match
                # there is the carry closing normally. Without this, the
                # unclosed branch below fires on every wrapped citation
                # that DID close and reports the partial fragment as a
                # dangling name: measured against the live wrap in
                # `tests/test_eval_cli.py`, not just against a fixture.
                carry_resolved = carry_resolved or i == 1
        # Only an identifier-shaped fragment continues; see the docstring.
        open_fragment = parts[-1] if len(parts) % 2 == 0 else ""
        identifier_shaped = re.fullmatch(r"[\w.]+", open_fragment)
        pending = open_fragment if identifier_shaped else None

        # A carry that neither closed nor continues was a citation missing
        # its second backtick. Reported at the line it opened on, which is
        # where a reader has to go to add the backtick.
        if carried is not None and pending is None and not carry_resolved:
            abandoned = _unclosed_citation(carried)
            if abandoned is not None:
                names.append((opened_at, abandoned))

    # The same event at the end of the text, where there is no next line to
    # abandon the carry on.
    if pending is not None:
        at_eof = _unclosed_citation(pending)
        if at_eof is not None:
            names.append((opened_at, at_eof))

    return names


def test_the_citation_scanner_resolves_the_shapes_a_citation_takes():
    """`_cited_names` against each shape, rather than against the tree.

    Split out of the repo-wide scan below: a failure in one of these
    fixtures is a bug in the scanner, and a failure in the scan below is a
    citation someone let rot. Reporting both under one name means the line
    pytest prints tells a reader neither which of the two happened nor
    which file to open.

    That argument is made elsewhere on this branch, and is made here again
    rather than assumed: `85ae9be` split the page-guard tripwire out of a
    test named for `scores.tsv` columns, this module's own docstring says
    why the repo-wide scan does not live in a module about eval adapters,
    and `test_mcqscore_still_names_the_test_that_runs_its_recipe` has its
    own name for the same reason. Listed, not counted -- the scan's
    docstring below says why, and this branch has got the count wrong
    every time it has written one.

    Fixtures rather than the live instances, because the tree containing a
    case today is a coincidence and the property is not. The wrapped
    citation in `tests/test_eval_cli.py` is one rewrap away from being a
    same-line one.
    """
    # Literal backticks are safe in THIS fixture and in no other one below,
    # for a reason worth stating rather than relying on: the name is a real
    # test, so when the scan below reads this file the citation resolves.
    # An invented name in literal backticks would be reported as dangling
    # -- that happened -- which is why every other fixture here builds its
    # delimiters with `chr(96)` instead.
    wrapped = (
        "    # and compared in `test_the_page_quotes_the_extraction_failur\n"
        "    # es_note_as_the_command_prints_it`, because this arm cannot.\n"
    )
    assert _cited_names(wrapped) == [
        (1, "test_the_page_quotes_the_extraction_failures_note_as_the_command_prints_it")
    ], "a citation split across two comment lines is not being rejoined"
    # That exact comparison is also what pins "reported ONCE": a carry that
    # closes and is ALSO reported as an abandoned fragment returns two
    # entries here. A separate `len(...) == 1` assertion was written for
    # that and deleted -- it restated this one more weakly, which is the
    # widening `test_mcqscore_still_names_the_test_that_runs_its_recipe`
    # records being corrected for.

    # The built-delimiter case itself. The family of instruments that read
    # their own source and misreport what they find is listed in
    # `test_no_citation_names_a_test_that_does_not_exist` below -- one
    # home and no ordinal, because the sentences in this tree that numbered
    # their own case did not agree with each other.
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

    # A comment continuation indented under its own marker rejoins too. The
    # version that took at most one space after the `#` embedded the rest
    # of the indent in the middle of the name and reported nothing, which
    # is the shape of every other bug on this function.
    padded_continuation = (
        f"# {tick}test_a_name_that_wraps_\n"
        f"#     here{tick} and more prose\n"
    )
    assert _cited_names(padded_continuation) == [(1, wrapped_name)], (
        "only one space is being stripped after a comment marker, so an "
        "indented continuation rejoins with whitespace inside the name"
    )

    # An unclosed citation reaches the caller rather than vanishing, at the
    # end of the text...
    unclosed_at_eof = f"# see {tick}test_never_closed\n"
    assert _cited_names(unclosed_at_eof) == [(1, "test_never_closed")], (
        "a citation left open at end of file is being dropped instead of "
        "reported"
    )

    # ...and MID-FILE, which is the case that is nearly always the real
    # one: a missing backtick has more file after it almost by definition.
    # Reported at the line it opened on, not the line the carry died on.
    # The first version reported only the end-of-text case and read that as
    # having closed the hole; measured here as `[]` before the branch above
    # existed.
    unclosed_mid_file = (
        f"# see {tick}test_never_closed\n"
        "# for the scanner\n"
    )
    assert _cited_names(unclosed_mid_file) == [(1, "test_never_closed")], (
        "a citation whose closing backtick never came is dropped unless it "
        "happens to be on the last line of the file"
    )

    # The recorded cost of the bounded carry, pinned so it stays a known
    # limit: a genuine multi-line code span resets parity, and a citation
    # on its closing line is dropped. If this ever starts finding the
    # citation, the carry was widened and `_cited_names`' trade paragraph
    # needs rewriting -- the fixture is here to make that loud either way.
    after_code_span = (
        f"# see {tick}MCQScore(total=1,\n"
        f"# attempted=1){tick} and {tick}test_on_one_line{tick} for why\n"
    )
    assert _cited_names(after_code_span) == [], (
        "the line closing a multi-line code span is no longer scanned at "
        "inverted parity -- good, but the docstring says otherwise"
    )

    # The third spelling of a module citation: a backticked FILENAME. The
    # bare stem resolves through `path.stem` and the `tests/...` path
    # through the regex in the scan below; this one matched neither, and the
    # tree's only instance was this file's own module docstring.
    with_extension = f"{tick}test_on_one_line.py{tick} and prose"
    assert _cited_names(with_extension) == [(1, "test_on_one_line")], (
        "a module cited as a backticked filename is invisible to the scan"
    )


def test_no_citation_names_a_test_that_does_not_exist():
    """Every `test_...` named in prose, across `src/` and `tests/`.

    Not "comment", which is the name this guard carried from `7210712`
    until the commit that renamed it. Most of what it resolves sits in
    DOCSTRINGS rather than comments -- the wrap handling in `_cited_names`
    exists because of them -- and a failure line naming a category
    narrower than what failed sends a reader looking for the wrong thing.
    The ratio is not written down here: a tally kept in prose is a claim
    with nowhere to check it, which is the argument the family list below
    makes, and this one would go stale the next time either text grows.

    `MCQScore`'s citation IS a comment, which is how the old name kept
    looking right: the one citation with a test of its own to be read
    beside it is in the minority spelling.

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
    - this docstring's first invented example, and the fixture name that
      replaced it
    - `_cited_names` itself, which rejoined a wrapped citation correctly
      and then looked for it at a parity the scan never reads
    - a sentence three functions down that said what a derivation bought
      and had the direction backwards, in a paragraph arguing for the
      derivation

    Its blind spots are worth stating for the same reason:

    - it treats any backticked `test_`-prefixed token as a claim that
      something exists
    - `known` is built from `^def test_` at column zero, so a test defined
      inside a class would be absent and every citation of it reported --
      loudly, which is the direction to prefer
    - the PATH spelling is matched per line, so a path reference that wraps
      is invisible where a wrapped backticked name is not. No instance
      today; the asymmetry is here because both checks now sit in one place
    - the line after a genuine multi-line code span is scanned at inverted
      parity, so a citation on it is dropped. That is the losing side of
      the bounded carry, reasoned about in `_cited_names` and pinned by a
      fixture -- a limit, not an oversight
    - a module named by BARE filename in prose, with no `tests/` prefix and
      no backticks, is not resolved. Deliberate and measured: dropping the
      prefix requirement reported two fixture filenames as dangling, and a
      fixture cannot avoid containing the shape it tests
    """
    root = Path(__file__).parent.parent
    tests_dir = root / "tests"

    known: set[str] = set()
    for path in sorted(tests_dir.rglob("*.py")):
        known.add(path.stem)
        text = path.read_text(encoding="utf-8")
        known.update(re.findall(r"^def (test_\w+)", text, re.M))

    # A path reference is a citation too, and the one in `src/` that the
    # backtick rule cannot reach: `MatrixConfig.on_scores` cites
    # tests/test_eval_matrix.py by path. The backtick requirement exists
    # because an unquoted NAME in prose is not distinguishable from a
    # sentence -- which is not true of a path.
    #
    # The `tests/` prefix is load-bearing, not incidental, and this was
    # measured rather than assumed: dropping it to cover a BARE filename in
    # prose -- `test_provider_errors.py` is cited that way inside
    # tests/test_biomni_provider.py -- immediately reported two FIXTURE
    # filenames as dangling: the decoy tree's module and this file's own
    # `.py` fixture. Neither is named in backticks here, because doing so
    # would make this comment a third false positive, which is the point it
    # is making. A fixture that exercises the scanner has to
    # contain the shape being scanned, and unlike the backticks it cannot
    # be built around: the token IS the match. The directory is what tells
    # a citation apart from test data, so an unprefixed filename stays a
    # stated blind spot rather than a false positive factory.
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

    Attribution resets at every top-level `def` AND every top-level
    `class`, for the reason `_page_guards` records: crediting a helper's
    text to the test above it is how that function came to count its own
    source. The `class` half was a live leak, not a precaution -- without
    it, 129 lines of three classes in `test_evaluation.py` are credited to
    the last test defined above them and 21 lines of a probe client in
    `test_falcon_errors.py` to another. Neither carries the one needle
    this is called with today, which is why nothing was failing and why
    the leak had to be measured rather than observed.

    A test defined INSIDE a class is not attributed at all, the same blind
    spot `known` has two functions up and for the same reason: the match is
    anchored at column zero.
    """
    found: set[str] = set()
    for path in sorted((Path(__file__).parent).rglob("*.py")):
        current: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"(?:def|class) (\w+)", line)
            if match:
                name = match.group(1)
                current = name if name.startswith("test_") else None
            elif needle in line and current:
                found.add(current)
    return found


def test_mcqscore_still_names_the_test_that_runs_its_recipe():
    """The citing direction, which the dangling check cannot see.

    `test_no_citation_names_a_test_that_does_not_exist` fails when a
    citation points at nothing. It stays green when a citation is simply
    DELETED -- and `MCQScore`'s class comment says naming its test is
    deliberate, and that describing the recipe without saying what runs it
    "is what left this comment unexecuted for four commits". A tidy-up that
    drops the name while keeping the recipe the other guards pin returns
    the comment to the state its own sentence calls the failure.

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
    `tests/` for the one whose body carries it. So no literal name is
    duplicated here to go stale, swapping the citation for an unrelated
    test fails, and RENAMING the recipe test fails here too -- the message
    names the test's new name and `models.py`'s citation is what has to
    change to match it. Importing the test would have been the other way
    round, and `tests/` is not an importable package.

    That last clause said "a rename of that test still passes (it is the
    same test)" from `f799e46` until the commit carrying this paragraph,
    which is backwards: the derivation is what makes a rename LOUD, and a
    hardcoded name is what would have made it quiet. The sentence
    described the alternative it was arguing against, in the paragraph
    arguing against it. Measured by renaming the recipe test and leaving
    `models.py` alone -- the assertion below fails, naming both sides.

    From outside that test, so deleting it cannot delete the checker.
    """
    # Imported inside the function, not at module scope: every other check
    # in this module reads the tree as TEXT and needs no eval package at
    # all, and a top-level import would make a repo-hygiene collection
    # error out of an unrelated import failure in `evaluation`.
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
        f"{sorted(runs_the_recipe)} -- with none, the recipe is documented "
        f"and unrun and there is no test for MCQScore to be naming; with "
        f"several, this check cannot say which one it should name"
    )
    expected = runs_the_recipe.pop()

    cited = [name for _, name in _cited_names(source)]
    assert expected in cited, (
        f"MCQScore's class comment no longer names {expected}, the test "
        f"that executes its documented recipe; the expression is still "
        f"pinned, but a reader has nothing to run it by. Cited: {cited}"
    )
