"""Load a fairseq ``.pt`` checkpoint without fairseq / omegaconf installed.

The official MusicBERT checkpoint pickles its ``cfg`` as an ``omegaconf``
``DictConfig`` and various ``fairseq`` dataclasses.  Those packages are not
dependencies of this project, so we install a temporary import hook that
fabricates harmless *placeholder* classes for anything under ``omegaconf`` /
``fairseq`` during unpickling.  ``numpy`` (which *is* installed) is left alone so
real arrays reconstruct normally.

Only :func:`load_fairseq_checkpoint` needs the hook; we uninstall it right after
loading so the rest of the process is unaffected.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from contextlib import contextmanager
from typing import Dict

import torch

_STUB_PREFIXES = ("omegaconf", "fairseq")


class _Placeholder:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["__pickle_state__"] = state

    def __call__(self, *a, **k):
        return self


class _FakeModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        cls = type(name, (_Placeholder,), {"__module__": self.__name__})
        setattr(self, name, cls)
        return cls


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in _STUB_PREFIXES and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _FakeModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


@contextmanager
def _stub_missing_modules():
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    # Remember which stub modules we create so we can clean them up.
    preexisting = set(sys.modules)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        for name in list(sys.modules):
            if name not in preexisting and name.split(".")[0] in _STUB_PREFIXES:
                if isinstance(sys.modules[name], _FakeModule):
                    del sys.modules[name]


def load_fairseq_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    """Return the full checkpoint dict (keys: model, cfg, args, ...).

    ``cfg`` / ``args`` may contain placeholder objects but ``model`` is a real
    ``OrderedDict`` of tensors.
    """
    with _stub_missing_modules():
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt


def load_fairseq_state_dict(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Return only the model ``state_dict`` (tensor names -> tensors)."""
    return load_fairseq_checkpoint(path, map_location=map_location)["model"]
