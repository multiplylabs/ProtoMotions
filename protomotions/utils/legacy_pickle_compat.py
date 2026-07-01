# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backward-compatibility shims for unpickling older resolved_configs.

``resolved_configs(_inference).pt`` pickles real Python objects (MdpComponent
``compute_func`` callables, config dataclasses) by their *import path*. When the
cart-pushing task code was reorganized — moved out of a top-level ``tasks``
package into ``protomotions.envs.*`` and some functions renamed — those old
import paths stopped resolving, so ``torch.load`` raised ``ModuleNotFoundError:
No module named 'tasks'``.

``register_legacy_module_aliases()`` installs ``sys.modules`` entries that map
the old ``tasks.*`` module paths to their current locations (handling the
function renames), so checkpoints trained before the refactor still load. Call
it once before ``torch.load`` of a resolved-configs pickle.
"""

import dataclasses
import importlib
import logging
import sys
import types

log = logging.getLogger(__name__)

# Substrings that mark a filesystem path baked in by an Anyscale/Ray cluster run.
# These point into the per-session Ray working dir, which never exists locally.
_RAY_PATH_MARKERS = (
    "/tmp/ray/",
    "runtime_resources/working_dir_files",
    "runtime_env_packages",
    # Stable per-worker symlink used by the seahorse robust_wbc Anyscale entrypoint
    # so asset paths survive retries (train_robust_wbc_anyscale.py). It points into
    # the materialized DVC assets on the cluster and never exists locally.
    "/tmp/seahorse_robust_wbc",
)

_MISSING = object()


def _is_baked_ray_path(value) -> bool:
    return isinstance(value, str) and any(m in value for m in _RAY_PATH_MARKERS)


def _field_default(field: "dataclasses.Field"):
    """Return a dataclass field's default value, or ``_MISSING`` if it has none."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return _MISSING


def sanitize_anyscale_paths(obj, _seen=None, _path="config") -> None:
    """Rewrite Anyscale/Ray-baked filesystem paths back to local defaults, in place.

    Checkpoints trained on an Anyscale/Ray cluster bake absolute
    ``/tmp/ray/session_.../working_dir_files/.../<asset>`` paths into config
    fields (robot ``asset_root``, ``scene_file``, ``motion_file``,
    cart-pushing ``init_pose_file``, ...). Those paths never exist locally.

    Any string dataclass field whose value looks like such a baked path is reset
    to that field's declared default — which is the canonical repo-relative path
    (e.g. ``protomotions/data/assets``). Fields that default to ``None``
    (``motion_file``/``scene_file``) are cleared so the inference CLI flags
    (``--motion-file`` / ``--scenes-file``) fill them. Recurses through nested
    dataclasses, dicts, lists and tuples; a no-op for locally-trained configs.
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if _is_baked_ray_path(value):
                default = _field_default(field)
                if default is _MISSING:
                    log.warning(
                        "Baked Ray path at %s.%s has no default; leaving as-is "
                        "(supply it via a CLI flag/override): %s",
                        _path,
                        field.name,
                        value,
                    )
                else:
                    setattr(obj, field.name, default)
                    log.info(
                        "Rebased Ray path %s.%s: %s -> %s",
                        _path,
                        field.name,
                        value,
                        default,
                    )
            else:
                sanitize_anyscale_paths(value, _seen, f"{_path}.{field.name}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if _is_baked_ray_path(value):
                log.warning(
                    "Baked Ray path at %s[%r] is a bare dict value with no "
                    "dataclass default; leaving as-is: %s",
                    _path,
                    key,
                    value,
                )
            else:
                sanitize_anyscale_paths(value, _seen, f"{_path}[{key!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            sanitize_anyscale_paths(value, _seen, f"{_path}[{i}]")

# Old fully-qualified module path -> current module path. The current module
# must expose the same attribute names the pickle references (see
# ``_RENAMED_ATTRS`` for paths where a symbol was also renamed).
_MODULE_ALIASES = {
    "tasks.cart_pushing.control": "protomotions.envs.control.cart_pushing_control",
    "tasks.cart_pushing.obs": "protomotions.envs.obs.cart_pushing",
    "tasks.cart_pushing.rewards": "protomotions.envs.rewards.cart_pushing",
    "tasks.common.historical_obs": "protomotions.envs.obs.humanoid_historical",
}

# Old module path -> {old_attr_name: new_attr_name} for symbols renamed during
# the refactor. The alias module gets the old names bound to the new objects.
_RENAMED_ATTRS = {
    "tasks.common.historical_obs": {
        "compute_masked_historical_max_coords_from_motion_lib": "compute_historical_max_coords_from_motion_lib",
        "compute_masked_historical_max_coords_from_state": "compute_historical_max_coords_from_state",
    },
}

# Parent packages that must exist for ``import tasks.cart_pushing.control`` to
# walk the dotted path. Order matters: parents before children.
_PARENT_PACKAGES = ["tasks", "tasks.cart_pushing", "tasks.common"]


def register_legacy_module_aliases() -> None:
    """Install ``sys.modules`` aliases so pre-refactor checkpoints unpickle.

    Idempotent and best-effort: a target module that no longer exists is logged
    and skipped rather than raising, so this never blocks loading a checkpoint
    that doesn't reference the missing path.
    """
    # Create lightweight package objects for the parents (with __path__ so they
    # behave like packages and attribute access on them works).
    for pkg_name in _PARENT_PACKAGES:
        if pkg_name in sys.modules:
            continue
        # Prefer a REAL package if it is importable (e.g. the seahorse `tasks`
        # package on PYTHONPATH, which holds current, non-renamed modules such as
        # ``tasks.common.terrain_conforming_obs``). Importing it keeps its real
        # ``__path__`` so those submodules still resolve during unpickling. Only
        # fall back to an empty stub when no real package exists — the true
        # pre-refactor case where ``tasks`` was removed entirely. The explicit
        # ``_MODULE_ALIASES`` below still override the 4 renamed legacy modules.
        try:
            importlib.import_module(pkg_name)
            continue
        except ImportError:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []  # marks it as a package
            sys.modules[pkg_name] = pkg

    for old_path, new_path in _MODULE_ALIASES.items():
        try:
            target = importlib.import_module(new_path)
        except ImportError as exc:
            log.warning(
                "Legacy alias %s -> %s skipped: %s", old_path, new_path, exc
            )
            continue

        renames = _RENAMED_ATTRS.get(old_path)
        if renames:
            # Build a dedicated alias module so we can bind the old attribute
            # names without mutating the real module's namespace.
            alias = types.ModuleType(old_path)
            alias.__dict__.update(target.__dict__)
            for old_attr, new_attr in renames.items():
                alias.__dict__[old_attr] = getattr(target, new_attr)
            module_obj = alias
        else:
            module_obj = target

        sys.modules[old_path] = module_obj
        # Wire the child onto its parent package as an attribute, so both
        # ``import tasks.cart_pushing.control`` and attribute traversal resolve.
        parent_name, _, child_name = old_path.rpartition(".")
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, module_obj)
