# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
"""Per-env terrain difficulty curriculum (IsaacLab ``terrain_levels_vel`` style).

Each env tracks a difficulty ``level`` (a terrain row). At reset, an env is promoted
to a harder level if it did well on its current one, or demoted if it failed; all envs
start low (``max_init_level``) so training ramps easy -> hard. The terrain lays out
rows by difficulty, and ``Terrain.sample_locations_for_levels`` spawns an env in its
level's band.

Promotion rule (mirrors the terrain-gated AMP: the human reference is *required* on
flat/low levels and *relaxed* on rough/high levels):

    promote = (distance > promote_distance) AND style_ok
    style_ok = (level >= style_required_below_level)         # rough: style not required
               OR (episode-mean AMP reward >= amp_bar)       # flat: must be human-like
    demote  = (distance < demote_distance) AND NOT promote

where ``distance`` is how far the robot traveled this episode and the episode-mean AMP
reward is the discriminator's human-likeness signal (fed in via ``record_amp``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TerrainCurriculumConfig:
    """Config for the per-env terrain difficulty curriculum."""

    enabled: bool = False
    max_init_level: int = 0  # envs start uniformly in [0, max_init_level]
    promote_distance: float = 3.0  # meters traveled to graduate to a harder level
    demote_distance: float = 0.5  # below this -> drop a level
    amp_bar: float = 0.5  # episode-mean AMP reward required to promote on a "flat" level
    style_required_below_level: int = 2  # levels < this require the AMP bar; rougher ones don't


class TerrainCurriculum:
    """Per-env terrain level state + the promote/demote decision (option B)."""

    def __init__(self, num_envs: int, num_levels: int, config: TerrainCurriculumConfig, device):
        self.num_envs = num_envs
        self.num_levels = num_levels
        self.cfg = config
        self.device = device
        max_init = min(config.max_init_level, num_levels - 1)
        self.terrain_levels = torch.randint(0, max_init + 1, (num_envs,), device=device, dtype=torch.long)
        self._start_xy = torch.zeros(num_envs, 2, device=device)
        self._amp_sum = torch.zeros(num_envs, device=device)
        self._amp_count = torch.zeros(num_envs, device=device)

    @torch.no_grad()
    def record_amp(self, amp_rewards: torch.Tensor) -> None:
        """Accumulate the per-step AMP (human-likeness) reward for the current episode."""
        self._amp_sum += amp_rewards.detach().to(self._amp_sum.dtype)
        self._amp_count += 1.0

    @torch.no_grad()
    def on_reset(self, env_ids: torch.Tensor, cur_xy: torch.Tensor) -> None:
        """Promote/demote the resetting envs from the episode that just ended.

        ``cur_xy`` is each env's current root xy (where the episode ended), measured
        BEFORE respawn. Updates ``terrain_levels`` in place and clears the per-episode
        accumulators for ``env_ids``. Call ``set_start_xy`` after respawn.
        """
        env_ids = env_ids.to(self.device).long()
        level = self.terrain_levels[env_ids]
        distance = torch.norm(cur_xy.to(self.device) - self._start_xy[env_ids], dim=-1)
        ep_amp = self._amp_sum[env_ids] / self._amp_count[env_ids].clamp(min=1.0)

        # Only adjust envs that actually ran an episode (skip the very first reset,
        # where no steps were recorded -> avoids spurious promotion/demotion).
        valid = self._amp_count[env_ids] > 0
        style_ok = (level >= self.cfg.style_required_below_level) | (ep_amp >= self.cfg.amp_bar)
        promote = (distance > self.cfg.promote_distance) & style_ok & valid
        demote = (distance < self.cfg.demote_distance) & ~promote & valid

        new_level = (level + promote.long() - demote.long()).clamp(0, self.num_levels - 1)
        self.terrain_levels[env_ids] = new_level

        self._amp_sum[env_ids] = 0.0
        self._amp_count[env_ids] = 0.0

    @torch.no_grad()
    def set_start_xy(self, env_ids: torch.Tensor, spawn_xy: torch.Tensor) -> None:
        """Record the (new) spawn xy so next episode's distance is measured from it."""
        env_ids = env_ids.to(self.device).long()
        self._start_xy[env_ids] = spawn_xy.to(self.device)

    def mean_level(self) -> float:
        return float(self.terrain_levels.float().mean().item())
