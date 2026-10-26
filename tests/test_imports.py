"""Every module under src/ imports (each in a fresh interpreter)."""

import pytest

from tests.import_audit import modules, try_import


@pytest.mark.slow
@pytest.mark.parametrize("module", modules())
def test_module_imports(module):
    _, error = try_import(module)
    assert error is None, error
