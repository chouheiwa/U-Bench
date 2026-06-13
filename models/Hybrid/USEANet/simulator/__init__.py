# ``simulator.py`` is created in Task A3; until then this guard keeps the
# package importable so submodules (e.g. ``degradations``) can be imported.
try:
    from .simulator import UltrasoundDegradationSimulator, SimOutput
except ImportError:
    pass

__all__ = ["UltrasoundDegradationSimulator", "SimOutput"]
