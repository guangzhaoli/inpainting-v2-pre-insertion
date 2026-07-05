"""Utilities package with lazy exports to avoid importing heavy dependencies on package import."""

from importlib import import_module


__all__ = [
    "parse_args",
]


def __getattr__(name):
    if name == "parse_args":
        module = import_module("src.utils.options")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
