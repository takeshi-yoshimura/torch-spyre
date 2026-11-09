# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""JSON-compatible descriptions of Spyre physical tensor layouts.

No checkpoint format, file I/O or transfer scheduling is defined here.
"""

from collections.abc import Mapping
from typing import Any


def layout_to_json(layout) -> dict:
    return {
        "device_size": list(layout.device_size),
        "stride_map": list(layout.stride_map),
        "device_dtype": layout.device_dtype.name,
        "element_arrangement": layout.element_arrangement.name,
    }


def layout_from_json(d: Mapping[str, Any]):
    from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout

    return SpyreTensorLayout(
        list(d["device_size"]),
        list(d["stride_map"]),
        DataFormats.__members__[d["device_dtype"]],
        ElementArrangement.__members__[d["element_arrangement"]],
    )
