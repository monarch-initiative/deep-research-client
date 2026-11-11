"""Reading a provider's declared requirements must stay cheap (issue #70).

The CLI reads `credential_env_var`, `registration_requirement` and
`stub_reason` straight off each provider class, so `providers` imports every
provider module just to list what could be configured. A module that imports
its SDK at module scope makes that cost the SDK's import time -- seconds for
`edison_client`, whose litellm dependency also fetches a price list over the
network on import. SDKs belong inside the methods that use them.

Run in a fresh interpreter: this test process has long since imported the
SDKs through other tests, so `sys.modules` here would prove nothing.
"""

import subprocess
import sys
import textwrap

import pytest

#: SDKs no provider class may drag in merely by being imported.
HEAVY_SDKS = ("openai", "edison_client", "litellm", "aviary")


@pytest.mark.parametrize("sdk", HEAVY_SDKS)
def test_loading_every_provider_class_imports_no_sdk(sdk):
    """Every class is loaded, then the SDK is looked for in sys.modules."""
    script = textwrap.dedent(
        f"""
        import sys
        from deep_research_client.client import PROVIDER_CLASS_PATHS, load_provider_class

        for name in PROVIDER_CLASS_PATHS:
            load_provider_class(name)
        print({sdk!r} in sys.modules)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "False", (
        f"importing the provider classes imported {sdk}; move that import into "
        f"the method that uses it"
    )
