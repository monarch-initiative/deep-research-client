"""Shared helper for the bundled-rubric coverage guards.

Two tests in two files parametrize over the subjects in
`gene_spot_checks.yaml` -- one loads them, one scores them -- and each
names its subjects by hand, so each can fall behind the file. The guard
that catches that is the same in both places.
"""

def parametrized_subjects(func, position: int) -> set[str]:
    """Subjects named in a test's `@pytest.mark.parametrize`, for a coverage guard.

    `pytestmark` holds every mark on the function, so reading `mark.args[1]`
    unconditionally raises IndexError the moment anyone adds a second mark --
    `@pytest.mark.integration`, say -- instead of failing with a message. Only
    parametrize marks are read, and only those with an argument list.

    Args:
        func: The parametrized test function.
        position: Index of the subject within each parametrize tuple.

    Returns:
        The non-None subjects named, as a set.
    """
    subjects: set[str] = set()
    for mark in getattr(func, "pytestmark", []):
        if mark.name != "parametrize" or len(mark.args) < 2:
            continue
        for case in mark.args[1]:
            value = case[position] if isinstance(case, tuple) else case
            if value is not None:
                subjects.add(value)
    return subjects
