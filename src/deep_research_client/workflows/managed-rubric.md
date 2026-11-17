# Research Report Rubric

The deliverables are files in `/mnt/session/outputs/`: the report `REPORT.md`,
the plan `PLAN.md`, and one evidence file per source in `citations/`. Judge
them by reading those files. Each evidence file is YAML frontmatter (`key`,
`identifiers`, `title`, `source_url`, `content`) followed by the source's text
as retrieved.

## Evidence

- Each evidence file's body is source text, not a summary written by the
  agent. Missing or inconsistent frontmatter fields (`key`, `content`) are not
  a reason to ask for revision: the file name stands in for a missing key.
- Identifiers in an evidence file are consistent with its body and
  `source_url` (for example, a PubMed URL whose PMID matches).

## Citations are traceable

- Every substantive factual claim in the report carries an inline citation: an
  identifier such as `(PMID:5568449)` or `(DOI:10.1093/ajcn/24.4.432)`, or a URL
  for a source without one.
- Every identifier cited in the report appears in the `identifiers` of an
  evidence file. An identifier with no evidence file is a failure: it was
  recalled, not retrieved.
- Every quotation written `"..." (ID)` appears in the body of the evidence file
  for that identifier, word for word.

## Citations are aligned

- Each cited source actually supports the sentence it is attached to: read the
  evidence file and check that it states what the report says it states. A
  citation to a real source that says something else is a failure, as is a
  number or claim stronger than the source's own.
- Each quotation supports, and is relevant to, the sentence it is attached to.
- Where sources disagree, the report says so rather than citing only one side.

## Sources are relevant

- The cited sources bear on the research question, and most claims rest on
  primary literature or authoritative reviews rather than general web pages.
- No section rests on a single source when the evidence files hold more.

## Structure

- Where the research question specifies sections, an output format or other
  content (such as ontology term suggestions), the report follows it, and the
  criteria below give way to it.
- Opens with an introduction stating the main findings, and states its
  interpretation if the question was ambiguous.
- Roughly six to eight `##` sections of narrative paragraphs in the style of a
  review article, not bullet lists (tables are fine where they help).
- An `## Open Questions` section saying what would be useful to follow up.
- A `## References` section with one line per cited source, giving its
  identifier, a full citation and its evidence file.

## Coverage

- Sample at least ten evidence files across the sub-questions and their
  findings cards (`<key>.findings.md`). Any specific finding there that matters
  to the research question (a number, a named model or variant, a negative or
  contrary result) and is missing from the report is a gap: name it.

- Each sub-question in `PLAN.md` is answered in the report or explicitly set
  aside as not covered, with the reason.
