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
#
"""Helpers for syncing training artifacts to Weights & Biases."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

CONFIG_FILENAMES = (
    "config.yaml",
    "resolved_configs.pt",
    "resolved_configs.yaml",
    "resolved_configs_inference.pt",
    "resolved_configs_inference.yaml",
    "experiment_config.py",
)

INFERENCE_CONFIG_FILENAMES = (
    "resolved_configs_inference.pt",
    "resolved_configs_inference.yaml",
)


def get_wandb_entity() -> Optional[str]:
    """Return W&B entity from env, or None to use the API key default."""
    return os.environ.get("WANDB_ENTITY") or None


def get_wandb_project(default: str = "physical_animation") -> str:
    """Return W&B project from env."""
    return os.environ.get("WANDB_PROJECT", default)


def wandb_run_is_active() -> bool:
    try:
        import wandb
    except ImportError:
        return False
    return wandb.run is not None


def upload_files_to_wandb(
    save_dir: Path,
    filenames: Iterable[str],
    *,
    artifact_type: str = "training-artifact",
    artifact_alias: Optional[str] = None,
) -> None:
    """Upload selected files from an experiment directory to the active W&B run.

    Files appear under the run's **Files** tab and as a versioned artifact when
    ``artifact_alias`` is set.
    """
    if not wandb_run_is_active():
        return

    import wandb

    save_dir = save_dir.resolve()
    existing_files = []
    for name in filenames:
        path = save_dir / name
        if path.is_file():
            existing_files.append(path)
        else:
            log.debug("Skipping W&B upload for missing file: %s", path)

    if not existing_files:
        return

    for path in existing_files:
        wandb.save(str(path), base_path=str(save_dir), policy="now")
        log.info("Uploaded to W&B: %s", path.relative_to(save_dir))

    if artifact_alias is not None:
        artifact_name = f"{save_dir.name}-{artifact_type}"
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        for path in existing_files:
            artifact.add_file(str(path), name=path.name)
        wandb.log_artifact(artifact, aliases=[artifact_alias])
        log.info(
            "Logged W&B artifact %s with alias %s (%d files)",
            artifact_name,
            artifact_alias,
            len(existing_files),
        )


def upload_configs_to_wandb(save_dir: Path) -> None:
    """Upload resolved training configs to the active W&B run."""
    upload_files_to_wandb(
        save_dir,
        CONFIG_FILENAMES,
        artifact_type="config",
        artifact_alias="configs",
    )


def upload_checkpoints_to_wandb(
    save_dir: Path,
    *checkpoint_names: str,
    epoch: Optional[int] = None,
) -> None:
    """Upload model checkpoint files to the active W&B run.

    Also uploads ``resolved_configs_inference.pt`` when present so a downloaded
    checkpoint directory is ready for ``inference_agent.py``.
    """
    alias = f"epoch-{epoch}" if epoch is not None else None
    filenames = list(checkpoint_names) + list(INFERENCE_CONFIG_FILENAMES)
    upload_files_to_wandb(
        save_dir,
        filenames,
        artifact_type="checkpoint",
        artifact_alias=alias,
    )
