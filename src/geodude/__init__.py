"""Geodude: High-level API for bimanual robot manipulation."""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies during testing
def __getattr__(name):
    if name == "Geodude":
        from geodude.robot import Geodude
        return Geodude
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Geodude"]
