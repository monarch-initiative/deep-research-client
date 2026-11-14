"""Check that the template commands documented in bundled skills actually render.

A skill is read by an agent that runs its commands as written, so a template
example with a wrong path or a missing ``--var`` fails in someone else's
project. These tests parse each ``research --template`` command out of the
skill's markdown and render it with the real template processor -- no
provider is called.
"""

import re
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from deep_research_client.processing import ResearchProcessor

SKILLS_DIR = Path(__file__).parent.parent / ".claude" / "skills"
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _template_commands(skill_md: Path) -> List[Tuple[Path, List[str]]]:
    """Return ``(template path, --var assignments)`` for each templated research command.

    ``SKILL_DIR`` in a documented path stands for the skill's own directory.
    Every template path must start with it: an installed skill runs from the
    user's project, so a path relative to this checkout would not exist there.
    Resolving such a path against the current directory would let it pass
    here (the repo root under ``just test``) while failing for every user.
    """
    commands = []
    for block in _BASH_BLOCK.findall(skill_md.read_text(encoding="utf-8")):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            if "--template" not in line:
                continue
            args = shlex.split(line, comments=True)
            template = args[args.index("--template") + 1]
            variables = [args[i + 1] for i, arg in enumerate(args) if arg == "--var"]
            assert template.startswith("SKILL_DIR/"), (
                f"{skill_md}: template path {template!r} must start with SKILL_DIR/ "
                f"so it resolves wherever the skill is installed"
            )
            path = skill_md.parent / template.removeprefix("SKILL_DIR/")
            commands.append((path, variables))
    return commands


def _skill_template_cases() -> Dict[str, Tuple[Path, List[str]]]:
    """Collect every templated command across bundled skills, keyed by a test id."""
    cases = {}
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        for n, command in enumerate(_template_commands(skill_md)):
            cases[f"{skill_md.parent.name}-{n}"] = command
    return cases


CASES = _skill_template_cases()


def test_run_deep_research_documents_a_template_command():
    """Guard the guard: the parser must find the example it is meant to check."""
    assert any(key.startswith("run-deep-research-") for key in CASES)


@pytest.mark.parametrize("template,variables", list(CASES.values()), ids=list(CASES))
def test_documented_template_command_renders(template: Path, variables: List[str]):
    """The template exists, every variable it uses is passed, and none is left unfilled."""
    processor = ResearchProcessor()

    valid, error = processor.validate_template_variables(template, variables)
    assert valid, error

    rendered = processor.template_processor.render_template(
        template, processor.template_processor.parse_variable_list(variables)
    )
    # Redundant for f-string templates, which already raise on a missing
    # variable; it catches what Jinja leaves behind. A template that needs a
    # literal "{word}" (JSON, say) would trip this and should relax it.
    assert not re.search(r"\{\w+\}", rendered), "unfilled placeholder left in rendered template"
