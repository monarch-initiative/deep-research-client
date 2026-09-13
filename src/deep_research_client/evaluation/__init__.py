"""Evaluation framework for deep research tools and co-scientists.

A benchmark is data here, not code. An adapter converts some upstream format
into an ``EvalSet`` of tasks, each declaring the shape of answer it expects, and
scorers dispatch on that shape rather than on the subject matter. Adding a
benchmark means writing one adapter class.

A run materialises results - for every task and arm, the prompt sent and the
response returned - and scoring is a separate step, so that a run stays useful
when the grading method changes.

Report scoring uses an LLM judge (FACT for citation support, RACE for report
quality, and claim recall against reference claims) plus intrinsic checks that
need no judge. Multiple-choice scoring is provisional; see ``mcq.py``.
"""
