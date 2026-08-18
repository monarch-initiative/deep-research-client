"""Topical relevance of a cited reference to the report that cites it.

Resolving an identifier proves the record exists. It does not prove the record has
anything to do with the report's subject, and an existence check cannot tell the
difference. A report on CHILD syndrome, a human sterol disorder, cited an
*Arabidopsis* pollen-development paper among its sources; that one turns out to
be a defensible citation of the plant orthologue, but nothing in the validator
was in a position to say so either way.

This module adds a cheap, transparent second opinion. It reads the report's own
distinctive vocabulary off the report, then asks how much of that vocabulary
appears in each reference's own metadata - title, journal, MeSH keywords and
abstract. Overlap is a clue, not a proof, and the vocabulary here is deliberately
conservative about saying so: only a reference with an abstract's worth of text
and almost no overlap is called ``OFF_TOPIC``, and everything ambiguous lands in
``UNCERTAIN`` rather than being scored against the report. That Arabidopsis paper
scores in the middle and is reported as ``UNCERTAIN``, which is the honest answer.

The module is dependency-free apart from the generated enum, so it can be
imported without the optional ``validation`` extra.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .datamodel import TopicalRelevance

__all__ = [
    "MIN_TEXT_FOR_A_NEGATIVE",
    "OFF_TOPIC_AT_OR_BELOW",
    "ON_TOPIC_AT_OR_ABOVE",
    "RelevanceAssessment",
    "ScoredTerm",
    "TopicalRelevance",
    "assess_relevance",
    "extract_keywords",
    "reference_text",
]

# How many of the report's terms the assessment is made against. Large enough
# that a paper on a sub-topic still matches several of them, small enough that
# the tail does not fill up with words that any biomedical paper would contain.
# Twenty separated best in the calibration described below; fifteen and thirty
# were close behind, so the exact figure is not delicate.
DEFAULT_KEYWORD_COUNT = 20

# A term mentioned once in a long report is as likely to be an aside as a theme.
MIN_TERM_OCCURRENCES = 2

# The two thresholds below are measured, not guessed, against two sets.
#
# The discrimination set pairs two real reports with their fetched bibliographies
# - a CHILD syndrome report with 34 references and an antiphospholipid syndrome
# report with 47 - giving 72 positives (each abstract against its own report) and
# 144 negatives (each abstract against a different report's keywords, including a
# third report on the Parkinson's gut microbiome). Every positive scored at or
# above 0.35; the negatives spread from 0.00 to 0.41.
#
# The false-accusation set is 2,561 references drawn from 400 curated Falcon
# disease reports, of which 2,025 carried enough text to judge. These are
# presumptively on topic, so anything flagged there is a cost. That set is what
# moved this threshold down: at 0.15 it would have flagged 5.6% of them, which is
# a false accusation roughly every other report.
#
# The two errors are not equally bad, which is why the thresholds sit well inside
# the gap rather than at its edges. Calling a good citation off topic is a false
# accusation printed in a user-facing report; missing a bad one only leaves
# things as they were before this check existed.

# Weighted overlap at or above this is positive evidence of topicality: it
# confirmed 72 of the 72 positives and only 4 of the 144 negatives.
ON_TOPIC_AT_OR_ABOVE = 0.35

# ...and at or below this, with enough text to judge from, negative evidence.
# Against 20 keywords this means matching one light term and nothing else. It
# still catches 27 of the 144 negatives, and flags 1.5% of the Falcon set -
# against 5.6% at 0.15, which was what three inspected false positives cost.
OFF_TOPIC_AT_OR_BELOW = 0.08

# Characters of *abstract* below which a low score says more about how little was
# retrieved than about the reference. A record that resolved to a title and
# nothing else is the common case - 536 of the 2,561 Falcon references, over a
# fifth - and a bare title cannot carry a fifth of a report's vocabulary however
# relevant the paper is. Those short records have a median score of 0.16 and a
# tenth of them score below 0.03, so without this gate they would dominate the
# flagged list while saying nothing.
#
# Measured on the abstract alone, not on everything searched: subject headings
# are searched, and are good evidence when they match, but they are controlled
# vocabulary and a paper can be squarely on topic while its MeSH terms share
# little with a report's prose. Letting a long heading list satisfy this gate
# would convict on exactly that mismatch.
MIN_TEXT_FOR_A_NEGATIVE = 300

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")

# The heading this project's own formatter puts above a provider's answer. The
# trailing anchor keeps it from matching the "## Output Format" heading that
# lives inside a template prompt.
_OUTPUT_HEADING = re.compile(r"^##[ \t]+Output[ \t]*$", re.MULTILINE)

# An author-year-title citation key, as Edison and Falcon write them inline:
# "(peduto2023neurofibromatosistype1 pages 1-2)". Matched by shape - a surname,
# a plausible publication year, then title words - rather than by "contains four
# digits", which was the first attempt and was wrong: KIAA0319, KIAA1109,
# KIAA0586 and KIAA0753 are real HGNC symbols. Dropping one costs a report its
# single most characteristic term, which then depresses the score of every
# reference that does discuss the gene - the very failure this rule exists to
# prevent, running backwards.
_CITATION_KEY = re.compile(r"[a-z]{2,}(?:19|20)\d{2}[a-z]{3,}")
_HEADING_SPLIT = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
# A word, optionally hyphenated (x-linked, post-squalene), starting with a
# letter so that bare numbers and identifiers do not become keywords.
_WORD = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")

# Splitting the document for the document-frequency term. Four is the point at
# which a heading split carries more information than the paragraph fallback.
_MIN_DOCUMENTS_FOR_HEADINGS = 4

# Shortest line that repeating makes suspicious. Below this, repetition is
# ordinary: "- **Gene:**", "| --- | --- |", a recurring bullet label.
_MIN_REPEATED_LINE = 25

# Function words, the boilerplate vocabulary of any research write-up, and the
# fragments that URL and citation syntax leaves behind. None of these can
# distinguish one report's subject from another's, which is the entire job here.
_STOPWORDS = frozenset(
    """
a about above across after again against all almost along already also although always among an
and another any anyone anything are around as at back be became because become becomes been before
began begin behind being below beside besides best better between beyond both but by came can
cannot certain clearly come common commonly could currently despite did different do does doing
done down due during each earlier early either else enough entire especially essentially even ever
every everything except far few finally first five following for four from full further gave
general generally get give given gives go goes going got greater had has have having he hence her
here hers herself high higher him himself his hold how however i if in indeed instead into is it
its itself just keep kept know known large larger last late later least left less let like likely
little long longer made mainly major make makes making many may maybe me mean means might more most
mostly much must my myself near nearly need needed neither never nevertheless new next no none nor
not nothing now number obtained of off often on once one only onto or order other others otherwise
our ours ourselves out over overall own part particular particularly per perhaps possible possibly
potentially predominantly present presented previously primarily probably quite range rather really
recent recently regarding relatively remain remains respectively same second see seem seems seen
several shall she short should show showed showing shown shows significant significantly similar
similarly since single six small so some sometimes somewhat specific still strong subsequently such
suggest suggested suggests take taken than that the their theirs them themselves then there thereby
therefore these they third this those though three through throughout thus to together too toward
towards two typically under unless until up upon us use used uses using usually various very well
were what when where whereas whether which while who whom whose why wide widely will with within
was without would yet you your yours

analysis approach approaches article articles assess assessed author authors background based
basis case cases conclusion conclusions consider considered context data date describe described
design detail details discussed discussion evidence example examples factor factors figure finding
findings framework further group groups importance important information introduction issue issues
journal key level levels limitation limitations literature main method methodology methods
objective objectives outcome outcomes overview paper papers point points population practice
preprint problem process publication published purpose question questions rate ratio reason
recommendation reference references report reported reporting reports research researcher
researchers result results review reviewed reviews role sample scope section sections setting
source sources studied studies study summary supplementary table tables term terms text title topic
trial trials understanding value values work

accessed available consulted database databases entry keyword keywords link links lookup query
querying record records resource resources retrieval retrieved search searched searches searching

applicable citation citations page pages

com doi edu europepmc geo gov href html http https medline mesh net nih nlm org pdf pmc pmcid pmid
pubmed www

cdc clinvar genereview icd omim orphanet statpearl uniprot
"""
    .split()
)


@dataclass(frozen=True)
class ScoredTerm:
    """One of a report's distinctive terms, with the weight it carries.

    Attributes:
        term: The term, lowercased and de-pluralised.
        score: Its TF-IDF weight, comparable only within one report.

    Examples:
        >>> ScoredTerm(term="ichthyosis", score=2.5).term
        'ichthyosis'
    """

    term: str
    score: float


@dataclass(frozen=True)
class RelevanceAssessment:
    """What the keyword overlap says about one reference.

    Attributes:
        relevance: The verdict, hedged towards ``UNCERTAIN``.
        score: Share of the report's keyword weight present in the reference.
        matched_terms: The report keywords that were found, heaviest first.

    Examples:
        >>> RelevanceAssessment(TopicalRelevance.UNCERTAIN, 0.1, ()).matched_terms
        ()
    """

    relevance: TopicalRelevance
    score: float
    matched_terms: tuple[str, ...]


def _strip_markup(text: str) -> str:
    """Remove the parts of a report that describe its plumbing, not its subject.

    Code fences, inline code and URLs are all rich in words that would otherwise
    dominate a frequency count - a report citing forty PubMed links is not a
    report about ``ncbi``.

    Args:
        text: Markdown or plain text.

    Returns:
        The text with fenced code, inline code and URLs blanked out.

    Examples:
        >>> _strip_markup("See https://pubmed.ncbi.nlm.nih.gov/1 for `NSDHL`.")
        'See   for  .'
    """
    text = _FENCED_CODE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    return _URL.sub(" ", text)


def _singular(word: str) -> str:
    """Fold the commonest English plural onto its singular.

    Enough morphology to stop ``lesions`` in a report missing ``lesion`` in an
    abstract, and no more: a real stemmer would be a dependency, and an
    aggressive one conflates terms that matter here (``sterol``/``steroid``).

    Args:
        word: A lowercased word.

    Returns:
        The word without a trailing plural ``s``.

    Examples:
        >>> _singular("lesions")
        'lesion'
        >>> _singular("analysis")
        'analysis'
        >>> _singular("bias")
        'bias'
        >>> _singular("cells")
        'cell'

        A doubled sibilant takes the whole ``es``, which the general rule would
        leave as ``processe``. It matters only for looks - both sides of a
        comparison are folded the same way either way - but these terms are
        printed in the report:

        >>> _singular("processes")
        'process'
        >>> _singular("diseases")
        'disease'
    """
    if len(word) > 5 and word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "as")):
        return word[:-1]
    return word


def _terms(text: str) -> list[str]:
    """Tokenise text into the terms relevance is judged on.

    Args:
        text: Markdown or plain text.

    Returns:
        Lowercased, de-pluralised, stopword-filtered terms, in order.

    Examples:
        >>> _terms("The ichthyosiform skin lesions are unilateral.")
        ['ichthyosiform', 'skin', 'lesion', 'unilateral']

        Gene and locus symbols are terms; the citation keys some providers write
        inline are not, and no abstract will ever contain one:

        >>> _terms("NF1 and POLR3A (peduto2023neurofibromatosistype1 pages 1-2)")
        ['nf1', 'polr3a']

        A digit-bearing symbol is a term, however many digits it carries:

        >>> _terms("KIAA0319 and KIAA1109 and CYP21A2")
        ['kiaa0319', 'kiaa1109', 'cyp21a2']
    """
    terms = []
    for word in _WORD.findall(_strip_markup(text).lower()):
        if len(word) < 3 or word in _STOPWORDS or _CITATION_KEY.match(word):
            continue
        singular = _singular(word)
        if singular in _STOPWORDS:
            continue
        terms.append(singular)
    return terms


def _topic_body(markdown: str) -> str:
    """Narrow a report to the part that states its findings.

    Two kinds of scaffolding are removed, because a templated prompt is
    word-for-word identical across every report made from it and its vocabulary
    - the databases to search first, the citation requirements, the output
    format - is both frequent and spread across every section, which is exactly
    the profile :func:`extract_keywords` rewards.

    First, the prompt itself: this project's formatter writes it under
    ``## Question`` and the provider's answer under ``## Output``.

    Second, whatever the provider echoed back. Falcon restates the entire prompt
    *inside* its own answer, so cutting at ``## Output`` removes only the first
    of two copies - and on a report with a short answer the surviving copy is
    most of the text. The echo is verbatim, which is the tell: a line that occurs
    twice in a report is scaffolding, and a sentence of findings repeated
    word for word is vanishingly rare. Short lines are exempt, since list items
    and table rules repeat for honest reasons.

    Reports with neither structure are returned unchanged, so this narrows the
    text when it can and never discards anything it does not recognise.

    Args:
        markdown: The report body.

    Returns:
        The findings, as far as they can be told apart from the ask.

    Examples:
        >>> _topic_body("## Question\\n\\nAsk about widgets.\\n\\n## Output\\n\\nBlue.\\n")
        'Blue.'
        >>> _topic_body("Just a report.")
        'Just a report.'

        A prompt's own "## Output Format" heading is not mistaken for it:

        >>> _topic_body("## Output Format\\n\\nUse a table.\\n")
        '## Output Format\\n\\nUse a table.'

        A prompt echoed back inside the answer goes too, both copies of it:

        >>> report = (
        ...     "## Question\\n\\nList every widget colour observed in the wild.\\n\\n"
        ...     "## Output\\n\\nList every widget colour observed in the wild.\\n\\n"
        ...     "Blue predominates.\\n"
        ... )
        >>> _topic_body(report)
        'Blue predominates.'
    """
    repeated = {
        line
        for line, count in Counter(
            stripped
            for stripped in (raw.strip() for raw in markdown.splitlines())
            if len(stripped) >= _MIN_REPEATED_LINE
        ).items()
        if count > 1
    }

    match = _OUTPUT_HEADING.search(markdown)
    body = markdown[match.end() :] if match else markdown

    return "\n".join(
        raw for raw in body.splitlines() if raw.strip() not in repeated
    ).strip()


def _documents(markdown: str) -> list[str]:
    """Split a report into the units the document-frequency term counts over.

    Sections first, because a report's headings are its natural divisions;
    paragraphs when there are too few headings for the split to mean anything.

    Args:
        markdown: The report body.

    Returns:
        The report's sections, or its paragraphs.

    Examples:
        >>> _documents("# A\\n\\nx\\n\\n# B\\n\\ny\\n\\n# C\\n\\nz\\n\\n# D\\n\\nw")
        ['A\\n\\nx\\n\\n', 'B\\n\\ny\\n\\n', 'C\\n\\nz\\n\\n', 'D\\n\\nw']

        Too few headings to divide the report, so paragraphs stand in:

        >>> _documents("# A\\n\\nx\\n\\n# B\\n\\ny")
        ['# A', 'x', '# B', 'y']
        >>> _documents("one para\\n\\ntwo para")
        ['one para', 'two para']
    """
    sections = [part for part in _HEADING_SPLIT.split(markdown) if part.strip()]
    if len(sections) >= _MIN_DOCUMENTS_FOR_HEADINGS:
        return sections
    return [part for part in _PARAGRAPH_SPLIT.split(markdown) if part.strip()]


def extract_keywords(
    markdown: str,
    top_n: int = DEFAULT_KEYWORD_COUNT,
) -> list[ScoredTerm]:
    """Read a report's distinctive vocabulary off the report itself.

    Terms are scored ``(1 + log10 tf) * (df / N)``: sublinear term frequency over
    the whole report, weighted by the share of the report's sections the term
    turns up in.

    That second factor is document frequency, not its inverse, and the direction
    is the whole point. Classic TF-IDF exists to find what distinguishes one
    document from a corpus; here there is no corpus, only one document about one
    subject, and a term's appearing in every section is the strongest evidence
    available that it *is* the subject. Inverting it was tried on a real 122-
    section report about CHILD syndrome and returned ``reproduced``, ``kegg``,
    ``rhea``, ``uspstf`` and ``atlas`` - each a single-section aside - while
    burying ``nsdhl``, ``cholesterol`` and ``ichthyosis``. Weighting by coverage
    instead returns ``child``, ``disease``, ``syndrome``, ``nsdhl``, ``skin``.

    Coverage also earns its keep against a stoplist's blind spot. A deep research
    report that documents where it searched mentions ``search`` more often than
    it mentions its own subject, but only inside its methods section, so coverage
    demotes it where raw frequency would have ranked it first.

    Coverage is also why the templated prompt has to go before any of this runs
    (see :func:`_topic_body`): prompt boilerplate is spread evenly across every
    section by construction, so it scores as though it were the subject.

    Args:
        markdown: The report body.
        top_n: How many terms to keep.

    Returns:
        The highest-scoring terms, heaviest first.

    Examples:
        >>> report = (
        ...     "## Skin\\n\\nThe ichthyosiform skin lesions stop at the midline.\\n\\n"
        ...     "## Sterols\\n\\nNSDHL blocks cholesterol synthesis; sterols accumulate.\\n\\n"
        ...     "## Therapy\\n\\nTopical cholesterol plus a statin clears skin lesions.\\n\\n"
        ...     "## Genetics\\n\\nNSDHL is X-linked, so lesions follow mosaic lines.\\n"
        ... )
        >>> [term.term for term in extract_keywords(report, top_n=4)]
        ['lesion', 'skin', 'cholesterol', 'nsdhl']

        A report with nothing to say yields no keywords, rather than a list of
        whatever it did contain:

        >>> extract_keywords("Nothing much here.")
        []
    """
    markdown = _topic_body(markdown)
    frequencies = Counter(_terms(markdown))
    documents = _documents(markdown)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(_terms(document)))
    total_documents = max(len(documents), 1)

    scored = [
        ScoredTerm(
            term=term,
            score=(1 + math.log10(count)) * (document_frequency[term] / total_documents),
        )
        for term, count in frequencies.items()
        if count >= MIN_TERM_OCCURRENCES
    ]
    # Ties broken alphabetically so the same report always yields the same list.
    scored.sort(key=lambda term: (-term.score, term.term))
    return scored[:top_n]


def reference_text(
    title: Optional[str] = None,
    content: Optional[str] = None,
    journal: Optional[str] = None,
    keywords: Optional[Iterable[str]] = None,
) -> str:
    """Assemble the reference metadata that relevance is judged against.

    MeSH keywords are included where a source supplies them: they are a
    professional indexer's answer to the question this function is asking, and
    they cost nothing extra to use.

    Args:
        title: Record title.
        content: Abstract or retrieved full text.
        journal: Journal or venue.
        keywords: Subject terms, such as MeSH headings.

    Returns:
        One block of text, or the empty string if the record carried none.

    Examples:
        >>> reference_text(title="A study", journal="J Widgets", keywords=["Widgets"])
        'A study\\nJ Widgets\\nWidgets'
        >>> reference_text()
        ''

        Subject terms are annotated by hand and cached as YAML, so a purely
        numeric one deserialises as an int. Coerced rather than trusted: this
        crashed a sweep over a real 37,000-record cache.

        >>> reference_text(title="A study", keywords=[2019])
        'A study\\n2019'
    """
    parts = [title, journal, *(keywords or []), content]
    return "\n".join(str(part) for part in parts if part)


def assess_relevance(
    keywords: Sequence[ScoredTerm],
    text: str,
    body: Optional[str] = None,
) -> RelevanceAssessment:
    """Judge one reference's metadata against a report's keywords.

    The score is the share of total keyword weight that appears in the text, so a
    reference that picks up the report's three heaviest terms scores higher than
    one that picks up its three lightest. Everything the record offers is
    searched - title, journal, subject headings and abstract alike.

    Verdicts are asymmetric on purpose. A high score is good evidence in one
    direction; a low score is only evidence at all when there was an abstract to
    have matched in, so a record that resolved to a title and a subject heading
    list is never called off topic no matter how little it shares.

    That gate is measured on ``body`` rather than on everything searched. A
    PubMed record with no abstract but a full MeSH list clears any length bar you
    care to set on the assembly, and would then be judged on controlled
    vocabulary that need not resemble the report's prose even when the paper is
    squarely on topic. Callers that pass only ``text`` get the safe reading: no
    body, so no accusation.

    Args:
        keywords: The report's keywords, from :func:`extract_keywords`.
        text: The reference metadata, from :func:`reference_text`.
        body: The abstract or retrieved full text alone, which is what decides
            whether a low score is worth acting on.

    Returns:
        The verdict, the score, and the keywords that matched.

    Examples:
        >>> keywords = [ScoredTerm("nsdhl", 3.0), ScoredTerm("cholesterol", 2.0),
        ...             ScoredTerm("lesion", 1.0)]
        >>> abstract = "NSDHL variants block cholesterol synthesis. " * 12
        >>> assessment = assess_relevance(keywords, abstract)
        >>> assessment.relevance == TopicalRelevance.ON_TOPIC
        True
        >>> assessment.matched_terms
        ('nsdhl', 'cholesterol')

        An abstract's worth of text, none of the report's vocabulary in it:

        >>> unrelated = "Pollen development in Arabidopsis thaliana anthers. " * 12
        >>> assess_relevance(keywords, unrelated, body=unrelated).relevance
        <TopicalRelevance.OFF_TOPIC: 'OFF_TOPIC'>

        The same words, but only a title's worth of them, is not enough to
        convict:

        >>> assess_relevance(keywords, "Pollen development in Arabidopsis").relevance
        <TopicalRelevance.UNCERTAIN: 'UNCERTAIN'>

        Nor is a title backed by a long list of subject headings, however much
        text that adds up to - none of it is the paper's own prose:

        >>> headings = "Pollen\\nArabidopsis\\nAnthers\\nPlant Infertility\\n" * 12
        >>> assess_relevance(keywords, headings, body="").relevance
        <TopicalRelevance.UNCERTAIN: 'UNCERTAIN'>

        With nothing to compare, nothing is claimed:

        >>> assess_relevance([], abstract).relevance
        <TopicalRelevance.NOT_ASSESSED: 'NOT_ASSESSED'>
        >>> assess_relevance(keywords, "").relevance
        <TopicalRelevance.NOT_ASSESSED: 'NOT_ASSESSED'>
    """
    total_weight = sum(keyword.score for keyword in keywords)
    if not total_weight or not text.strip():
        return RelevanceAssessment(TopicalRelevance.NOT_ASSESSED, 0.0, ())

    present = set(_terms(text))
    matched = [keyword for keyword in keywords if keyword.term in present]
    score = sum(keyword.score for keyword in matched) / total_weight

    if score >= ON_TOPIC_AT_OR_ABOVE:
        verdict = TopicalRelevance.ON_TOPIC
    elif score <= OFF_TOPIC_AT_OR_BELOW and len(body or "") >= MIN_TEXT_FOR_A_NEGATIVE:
        verdict = TopicalRelevance.OFF_TOPIC
    else:
        verdict = TopicalRelevance.UNCERTAIN

    return RelevanceAssessment(
        relevance=verdict,
        score=round(min(score, 1.0), 3),
        matched_terms=tuple(keyword.term for keyword in matched),
    )
