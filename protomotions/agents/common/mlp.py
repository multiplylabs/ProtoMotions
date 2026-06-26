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
"""Multi-Layer Perceptron (MLP) network implementations.

This module provides MLP architectures used throughout the codebase.
All MLPs support optional observation normalization and operate on TensorDict inputs.

These are the building blocks for actor and critic networks in RL agents.

Key Classes:
    - MLP: Feedforward network with optional observation normalization
    - MultiHeadedMLP: MLP with multiple parallel input heads and a trunk

Functions:
    - build_mlp: Factory function to construct MLP from configuration
"""

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.common.common import NormObsBase, apply_module_operations
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
)


def build_mlp(config: MLPWithConcatConfig):
    """Build a multi-layer perceptron from configuration using LazyLinear.

    Uses LazyLinear for automatic input size inference. The first forward pass
    will materialize the layers with the correct input dimensions.

    Args:
        config: MLP configuration specifying layers, activations, and output dimensions.

    Returns:
        Sequential neural network module with lazy initialization.
    """
    layers = []
    for i, layer in enumerate(config.layers):
        # Use LazyLinear - input size inferred on first forward
        layers.append(nn.LazyLinear(layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(get_activation_func(layer.activation))

    # Final layer also uses LazyLinear
    layers.append(nn.LazyLinear(config.num_out))
    return nn.Sequential(*layers)


class MLPWithConcat(TensorDictModuleBase):
    """Multi-layer perceptron network with optional observation normalization.

    Feedforward network that processes observations through multiple
    fully-connected layers with configurable activations. Optionally
    normalizes inputs using running mean/std statistics.

    REQUIRES explicit obs_key and out_key to prevent key collisions.
    Always operates on TensorDict for clean, traceable data flow.

    Args:
        config: Configuration specifying input/output dimensions, hidden layers,
               and normalization settings. Both obs_key and out_key must be explicitly set.

    Attributes:
        norm: NormObsBase module for optional normalization (plain nn.Module).
        mlp: Sequential network of linear layers and activations.
        in_keys: List of input keys (always non-empty).
        out_keys: List of output keys (always non-empty).
    """

    config: MLPWithConcatConfig

    def __init__(self, config: MLPWithConcatConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        # Validate TensorDict keys
        assert config.in_keys, "MLP requires obs_key to be explicitly set."
        assert config.out_keys, "MLP requires out_key to be explicitly set."

        # Create normalization module (plain nn.Module with lazy init)
        self.norm = NormObsBase(config)
        self.mlp = build_mlp(self.config)

        self.output_activation = None
        if self.config.output_activation is not None:
            self.output_activation = get_activation_func(self.config.output_activation)

        # Set up TensorDict keys
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        assert len(self.out_keys) == 1, "MLP requires exactly one output key"

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass with optional normalization.

        Args:
            tensordict: TensorDict containing observations.
            return_norm_obs: If True, stores normalized obs in tensordict.

        Returns:
            TensorDict with processed outputs.
        """
        combined_obs = torch.cat(
            [tensordict[key] for key in self.config.in_keys], dim=-1
        )

        result = apply_module_operations(
            combined_obs, self.config.module_operations, normalizer=self.norm, forward_model=self.mlp
        )

        outs = result["output"]

        if self.output_activation is not None:
            outs = self.output_activation(outs)

        tensordict[self.config.out_keys[0]] = outs
        if self.config.normalize_obs and result["norm_obs"] is not None:
            norm_obs = result["norm_obs"]
            # Only store if batch dimension matches (reshape operations may change it)
            if norm_obs.shape[0] == tensordict.batch_size[0]:
                tensordict[f"norm_{self.config.in_keys[0]}"] = norm_obs

        return tensordict


class ExtrinsicsEncoderMLP(MLPWithConcat):
    """MLPWithConcat actor trunk with an internal RMA-style extrinsics encoder.

    Splits its inputs into deployable proprioception (``config.in_keys``,
    normalized like the base MLP) and privileged domain-randomization factors
    (``config.extrinsics_keys``, e.g. terrain heightmap / friction / push). The
    privileged factors are encoded by a small MLP into a latent ``z_t`` of size
    ``config.extrinsics_latent_dim``, concatenated with the normalized
    proprioception, and fed to the trunk.

    The encoder lives inside this (actor) module, so it is optimized by the actor
    optimizer and DDP-wrapped together with the actor -- no agent/model changes.
    Used for RMA Phase-1 ("teacher"): z_t comes from privileged factors. Phase 2
    would replace the encoder with an adaptation module predicting z_t from
    proprioceptive history.
    """

    def __init__(self, config):
        super().__init__(config)
        assert config.extrinsics_keys, "ExtrinsicsEncoderMLP requires extrinsics_keys."
        enc_layers = []
        for layer in config.encoder_layers:
            enc_layers.append(nn.LazyLinear(layer.units))
            enc_layers.append(get_activation_func(layer.activation))
        enc_layers.append(nn.LazyLinear(config.extrinsics_latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Encode privileged extrinsics -> latent z_t (not obs-normalized).
        priv = torch.cat([tensordict[key] for key in self.config.extrinsics_keys], dim=-1)
        z = self.encoder(priv)

        # Normalize deployable proprioception (same running-norm as MLPWithConcat).
        proprio = torch.cat([tensordict[key] for key in self.config.in_keys], dim=-1)
        proprio = self.norm(proprio)

        outs = self.mlp(torch.cat([proprio, z], dim=-1))
        if self.output_activation is not None:
            outs = self.output_activation(outs)

        tensordict[self.config.out_keys[0]] = outs
        if self.config.normalize_obs:
            tensordict[f"norm_{self.config.in_keys[0]}"] = proprio
        return tensordict
