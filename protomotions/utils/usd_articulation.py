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
import os
from typing import Optional, Tuple


def usd_compute_bounding_box(
    usd_path: str,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Return (min_x, max_x, min_y, max_y, min_z, max_z) from a USD stage."""
    if not os.path.isfile(usd_path):
        return None

    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError:
        return None

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception:
        return None

    if stage is None:
        return None

    bounds = Gf.BBox3d()
    has_geometry = False
    for prim in stage.Traverse():
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        prim_bounds = imageable.ComputeWorldBound(
            Usd.TimeCode.Default(), UsdGeom.Tokens.default_
        )
        if prim_bounds.GetRange().IsEmpty():
            continue
        bounds = Gf.BBox3d.Combine(bounds, prim_bounds)
        has_geometry = True

    if not has_geometry:
        return None

    box_range = bounds.GetRange()
    min_point = box_range.GetMin()
    max_point = box_range.GetMax()
    return (
        float(min_point[0]),
        float(max_point[0]),
        float(min_point[1]),
        float(max_point[1]),
        float(min_point[2]),
        float(max_point[2]),
    )


def usd_has_articulation_root(usd_path: str) -> bool:
    """Return True when a USD file contains an articulation root prim."""
    if not os.path.isfile(usd_path):
        return False

    if usd_path.endswith(".usda"):
        try:
            with open(usd_path, encoding="utf-8") as usd_file:
                header = usd_file.read(65536)
            return "PhysicsArticulationRootAPI" in header
        except OSError:
            return False

    try:
        from pxr import Usd
    except ImportError:
        return False

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception:
        return False

    if stage is None:
        return False

    for prim in stage.Traverse():
        if prim.HasAPI("PhysicsArticulationRootAPI"):
            return True
    return False
