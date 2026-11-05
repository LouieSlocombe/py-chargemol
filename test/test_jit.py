"""Optional acceleration stays lazy and leaves NumPy-only installs usable."""

import inspect
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from py_chargemol import _jit


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_acceleration_requires_numba_and_respects_opt_out(
    monkeypatch: pytest.MonkeyPatch, installed: bool, disabled: bool
) -> None:
    monkeypatch.setattr(
        _jit, "find_spec", lambda _name: object() if installed else None
    )
    if disabled:
        monkeypatch.setenv("PY_CHARGEMOL_DISABLE_JIT", "1")
    else:
        monkeypatch.delenv("PY_CHARGEMOL_DISABLE_JIT", raising=False)
    assert _jit._enabled() == (installed and not disabled)  # noqa: SLF001


def test_disabled_decorator_retains_python_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_jit, "ENABLED", False)

    def identity(value: int) -> int:
        return value

    assert _jit.jit(identity) is identity


def test_compilation_is_lazy_cached_and_uses_strict_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    options: list[dict[str, bool]] = []

    def compiler(**kwargs: bool) -> Callable[[Callable[..., int]], Callable[..., int]]:
        options.append(kwargs)
        return lambda function: function

    def load(name: str) -> SimpleNamespace:
        imports.append(name)
        return SimpleNamespace(njit=compiler)

    monkeypatch.setattr(_jit, "ENABLED", True)
    monkeypatch.setattr(_jit, "import_module", load)

    @_jit.jit
    def add(left: int, *, right: int) -> int:
        return left + right

    assert not imports
    assert not options
    assert inspect.unwrap(add) is not add
    assert add(2, right=3) == 5
    assert add(3, right=4) == 7
    assert imports == ["numba"]
    assert options == [{"cache": True, "nogil": True, "fastmath": False}]
