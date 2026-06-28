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
"""G1 (29-dof body) + actuated Unitree Dex3 hands (7-dof/hand) = 43-dof robot.

Identical to ``G1RobotConfig`` for the body; the validated ``g1_holo_compat``
chain is reused verbatim. The two ``*_rubber_hand`` end-effectors are replaced by
the Dex3 fingers (merged into ``g1_holo_compat_dex3`` USD/MJCF). The 14 finger
DOFs (thumb_0/1/2, middle_0/1, index_0/1 per hand) are position-controlled.

The asset's kinematic_info is parsed from the MJCF, giving 43 dofs in tree order:
body dofs at indices {0-21, 29-35}, left fingers {22-28}, right fingers {36-42}.
The hand "body" is ``*_wrist_yaw_link`` (the Dex3 palm is rigid to it; there is no
separate rubber-hand body anymore).
"""
from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)

# Reuse the validated G1 body control constants + default standing pose.
from protomotions.robot_configs.g1 import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DEFAULT_JOINT_POS,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.isaacgym.config import IsaacGymPhysXParams, IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabPhysXParams, IsaacLabSimParams
from protomotions.simulator.newton.config import NewtonSimParams

# Dex3 finger PD gains. The fingers are small and position-controlled to discrete
# open/close presets, so modest gains suffice. Effort limits follow the Dex3 MJCF
# actuatorfrcrange (thumb_0 = 2.45 Nm, the rest 1.4 Nm). Tune if grasps slip.
ARMATURE_FINGER = 0.001
STIFFNESS_FINGER = 3.0
DAMPING_FINGER = 0.1

# Body default pose (palms-down handled by the init-pose .pt) + fingers open (0).
DEFAULT_JOINT_POS_DEX3 = {**DEFAULT_JOINT_POS, ".*_hand_.*_joint": 0.0}


@dataclass
class G1Dex3RobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            # No rubber-hand body on the Dex3 robot; the palm is rigid to wrist_yaw.
            "all_left_hand_bodies": ["left_wrist_yaw_link"],
            "all_right_hand_bodies": ["right_wrist_yaw_link"],
            "head_body_name": ["head"],
            "torso_body_name": ["torso_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso_link",
            "head",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
    )

    default_root_height: float = 0.8
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS_DEX3)
    anchor_body_name: str = "torso_link"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/g1_holo_compat_dex3.xml",
            usd_asset_file_name="usd/g1_holo_compat_dex3/g1_holo_compat_dex3.usdc",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                # --- G1 body (identical to G1RobotConfig) ---
                ".*_hip_(pitch|yaw)_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    effort_limit=88,
                    velocity_limit=32,
                    armature=ARMATURE_7520_14,
                ),
                ".*_hip_roll_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    effort_limit=139,
                    velocity_limit=20,
                    armature=ARMATURE_7520_22,
                ),
                ".*_knee_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    effort_limit=139,
                    velocity_limit=20,
                    armature=ARMATURE_7520_22,
                ),
                ".*_ankle_.*": ControlInfo(
                    stiffness=2 * STIFFNESS_5020,
                    damping=2 * DAMPING_5020,
                    effort_limit=50,
                    velocity_limit=37,
                    armature=2 * ARMATURE_5020,
                ),
                "waist_yaw_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    effort_limit=88,
                    velocity_limit=32,
                    armature=ARMATURE_7520_14,
                ),
                "waist_(roll|pitch)_joint": ControlInfo(
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    effort_limit=50,
                    velocity_limit=37,
                    armature=2.0 * ARMATURE_5020,
                ),
                ".*_(shoulder|elbow)_.*": ControlInfo(
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    effort_limit=25,
                    velocity_limit=37,
                    armature=ARMATURE_5020,
                ),
                ".*_wrist_roll_joint": ControlInfo(
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    effort_limit=25,
                    velocity_limit=37,
                    armature=ARMATURE_5020,
                ),
                ".*_wrist_pitch_joint": ControlInfo(
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    effort_limit=5,
                    velocity_limit=22,
                    armature=ARMATURE_4010,
                ),
                ".*_wrist_yaw_joint": ControlInfo(
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    effort_limit=5,
                    velocity_limit=22,
                    armature=ARMATURE_4010,
                ),
                # --- Dex3 fingers (14 dofs). Mutually-exclusive patterns. ---
                ".*_hand_thumb_0_joint": ControlInfo(
                    stiffness=STIFFNESS_FINGER,
                    damping=DAMPING_FINGER,
                    effort_limit=2.45,
                    velocity_limit=10,
                    armature=ARMATURE_FINGER,
                ),
                ".*_hand_(thumb_1|thumb_2|index_0|index_1|middle_0|middle_1)_joint": ControlInfo(
                    stiffness=STIFFNESS_FINGER,
                    damping=DAMPING_FINGER,
                    effort_limit=1.4,
                    velocity_limit=10,
                    armature=ARMATURE_FINGER,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=100,
                decimation=2,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=100,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
            ),
        )
    )
