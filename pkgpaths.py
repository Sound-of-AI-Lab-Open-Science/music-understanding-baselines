"""Python half of the package's one path configuration.

``env.sh`` and this module read the same ``paths.yaml`` and apply the same
rule: an environment variable that is already set wins, otherwise the default
from ``paths.yaml`` is used, with ``${...}`` references expanded against the
keys resolved before it. So a script run inside a job that sourced ``env.sh``
and the same script run by hand from a shell that did not both see identical
paths.

Usage, from a file three levels below the package root -- which is where every
current caller sits (``eval/benchmir/scripts_*/``)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # package root
    from pkgpaths import PATHS

    cfg = PATHS.benchmir_data_root / "cipi"

``parents[3]`` is the depth, not a constant: from anywhere else, count the
levels, or walk up to the directory that holds ``paths.yaml``::

    root = Path(__file__).resolve()
    while not (root / "paths.yaml").is_file():
        root = root.parent
    sys.path.insert(0, str(root))

Nothing here touches the filesystem; ``PATHS.prepare()`` is the one call that
creates directories, and only the entry points make it.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

#: The package root: the directory holding this file, env.sh and paths.yaml.
PKG_ROOT = Path(__file__).resolve().parent

_KV = re.compile(r'^([a-z_]+):\s*(.*?)\s*$')

#: paths.yaml key -> environment variable, in resolution order. A key may
#: reference any key above it, plus ${HOME} and ${PKG_ROOT}.
_ORDER = [
    ("data_root", "DATA_ROOT"),
    ("benchmir_data_root", "BENCHMIR_DATA_ROOT"),
    ("work_root", "WORK_ROOT"),
    ("checkpoint_root", "CHECKPOINT_ROOT"),
    ("conda_root", "CONDA_ROOT"),
    ("jepa_env", "JEPA_ENV"),
    ("musetok_env", "MUSETOK_ENV"),
    ("benchmir_env", "BENCHMIR_ENV"),
    ("benchmir_root", "BENCHMIR_ROOT"),
    ("musetok_repo", "MUSETOK_REPO"),
    ("slurm_partition", "JR_SLURM_PARTITION"),
    ("slurm_gpu_partition", "JR_SLURM_GPU_PARTITION"),
    ("slurm_account", "JR_SLURM_ACCOUNT"),
    ("slurm_qos", "JR_SLURM_QOS"),
    ("slurm_exclude", "JR_SLURM_EXCLUDE"),
    ("slurm_gres", "JR_SLURM_GRES"),
]

#: Derived locations, resolved after the table above. Same override rule.
_DERIVED = [
    ("CACHE_ROOT", "${WORK_ROOT}/cache"),
    ("SPLIT_ROOT", "${WORK_ROOT}/splits"),
    ("SPLIT_TAG", "main"),
    ("SPLIT_DIR", "${SPLIT_ROOT}/${SPLIT_TAG}"),
    ("VOCAB_ROOT", "${WORK_ROOT}/vocab"),
    ("RUN_ROOT", "${WORK_ROOT}/runs"),
    ("JOBS_ROOT", "${WORK_ROOT}/jobs"),
    ("LOG_ROOT", "${WORK_ROOT}/logs"),
    ("MIDI_DIR", "${DATA_ROOT}/musescore"),
    ("BENCHMIR_OURS_CACHE", "${WORK_ROOT}/cache/embeddings"),
    ("CONDA_ENVS_ROOT", "${CONDA_ROOT}/envs"),
    ("JEPA_ROOT", str(PKG_ROOT / "pretrain")),
    ("JEPA_CKPT", "${CHECKPOINT_ROOT}/music_jepa_final.pt"),
    ("MUSICBERT_CKPT", "${CHECKPOINT_ROOT}/musicbert_base_converted.pt"),
    ("MUSETOK_CKPT", "${MUSETOK_REPO}/ckpt/best_tokenizer/model.pt"),
    ("MUSETOK_MUSESCORE_VOCAB", "${VOCAB_ROOT}/dictionary_musescore.pkl"),
]

#: Names whose value is a name, not a path: never turned into a Path.
_PLAIN = {
    "SPLIT_TAG", "JEPA_ENV", "MUSETOK_ENV", "BENCHMIR_ENV", "JR_SLURM_PARTITION",
    "JR_SLURM_GPU_PARTITION", "JR_SLURM_ACCOUNT", "JR_SLURM_QOS", "JR_SLURM_EXCLUDE",
    "JR_SLURM_GRES",
}


def _read_defaults() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (PKG_ROOT / "paths.yaml").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = _KV.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip('"')
    return out


def _conda_root() -> str:
    """Prefix of the conda installation actually in use.

    ``conda_root`` is empty in ``paths.yaml`` by default, so this derives it
    from the running installation, in order: the base install ``$CONDA_EXE``
    points into; ``$CONDA_PREFIX`` (its grandparent when the active environment
    sits inside an ``envs/`` directory); ``conda`` on ``$PATH``; finally a probe
    of conventional install prefixes, because a batch job often has none of the
    above. Set ``$CONDA_ROOT`` if none of this finds yours.
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return str(Path(conda_exe).resolve().parent.parent)
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        return str(p.parent.parent if p.parent.name == "envs" else p)
    found = shutil.which("conda")
    if found:
        return str(Path(found).resolve().parent.parent)
    home = Path.home()
    candidates = sorted(home.glob("mini*3")) + sorted(home.glob("ana*3"))
    candidates += [Path("/opt/conda"), Path("/usr/local/conda")]
    for cand in candidates:
        if (cand / "bin" / "conda").is_file():
            return str(cand)
    return ""


class _Paths:
    """Every configured value, as attributes in lower case.

    ``PATHS.work_root`` is a ``Path``; ``PATHS.benchmir_env`` is a ``str``.
    Resolution also writes each value back into ``os.environ``, so a
    subprocess -- a model worker in another conda environment, say -- inherits
    exactly what this process resolved.
    """

    def __init__(self) -> None:
        defaults = _read_defaults()
        os.environ.setdefault("PKG_ROOT", str(PKG_ROOT))
        for key, var in _ORDER:
            self._resolve(var, defaults.get(key, ""))
        if not os.environ.get("CONDA_ROOT"):
            self._resolve("CONDA_ROOT", _conda_root())
        for var, default in _DERIVED:
            self._resolve(var, default)
        self.pkg_root = PKG_ROOT

    def _resolve(self, var: str, default: str) -> None:
        value = os.environ.get(var)
        if value is None or value == "":
            value = os.path.expandvars(default)
        os.environ[var] = value
        setattr(self, var.lower(), value if var in _PLAIN else Path(value))

    def python_for(self, env_name: str) -> Path:
        """Absolute interpreter for a conda environment name."""
        return Path(os.environ["CONDA_ENVS_ROOT"]) / env_name / "bin" / "python"

    def prepare(self) -> None:
        """Create the writable tree. Called by the entry points, not on import."""
        for d in (self.work_root, self.cache_root, self.split_root,
                  self.vocab_root, self.run_root, self.jobs_root,
                  self.log_root, self.checkpoint_root,
                  self.benchmir_ours_cache):
            d.mkdir(parents=True, exist_ok=True)


PATHS = _Paths()


def expandvars_tree(node):
    """Expand ``${VAR}`` in every string of a loaded YAML/JSON tree.

    The evaluation configs name ``${BENCHMIR_DATA_ROOT}`` and
    ``${CHECKPOINT_ROOT}`` rather than absolute paths. Job generators expand
    them once, when they write a per-job config, so the generated file records
    the concrete path it will be run against -- a pass whose job configs
    disagree about which corpus or checkpoint they used is not a pass.

    ``os.path.expandvars`` leaves an undefined variable verbatim, which then
    fails as a missing file instead of silently resolving to something else.
    """
    if isinstance(node, str):
        return os.path.expandvars(node)
    if isinstance(node, dict):
        return {k: expandvars_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [expandvars_tree(v) for v in node]
    return node


def write_lines(path, items) -> int:
    r"""Write one item per line, and an EMPTY file for an empty list.

    ``"\n".join([]) + "\n"`` is ``"\n"`` -- a single blank line, which ``wc -l``
    counts as one. The runners size every sbatch array by running ``wc -l`` on
    exactly these files, so an empty half of a grid became a phantom one-task
    array that started, read a blank line, printed "no job at line 1" and
    exited: ``run_eval.sh`` announced "eval heavy: 1 jobs" for a grid whose
    generator had just printed "0 heavy". Counting is only honest if the file
    is honest.

    Returns the number of items written, so a caller can report the same number
    the runner will later count.
    """
    items = list(items)
    Path(path).write_text("".join("{}\n".format(i) for i in items))
    return len(items)


def package_root() -> Path:
    """The package root, for scripts that need it without importing PATHS."""
    return PKG_ROOT


if __name__ == "__main__":  # `python pkgpaths.py` prints the resolved settings
    for name in sorted(vars(PATHS)):
        print(f"{name:24s} {getattr(PATHS, name)}")
