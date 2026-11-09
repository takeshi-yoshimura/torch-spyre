# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Layout metadata helpers tested without loading the Spyre runtime."""

import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def layouts(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "torch_spyre" / "layout.py"
    spec = importlib.util.spec_from_file_location("layout_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DataFormats(Enum):
        FP16 = 1

    class ElementArrangement(Enum):
        CONTIGUOUS = 1

    def layout(size, stride, dtype, arrangement):
        return SimpleNamespace(
            device_size=size,
            stride_map=stride,
            device_dtype=dtype,
            element_arrangement=arrangement,
        )

    runtime = ModuleType("torch_spyre._C")
    runtime.DataFormats = DataFormats
    runtime.ElementArrangement = ElementArrangement
    runtime.SpyreTensorLayout = layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    return module, layout([2, 4], [4, 1], DataFormats.FP16, ElementArrangement.CONTIGUOUS)


def test_layout_round_trip(layouts):
    module, layout = layouts
    encoded = json.loads(json.dumps(module.layout_to_json(layout)))
    assert encoded == {
        "device_size": [2, 4],
        "stride_map": [4, 1],
        "device_dtype": "FP16",
        "element_arrangement": "CONTIGUOUS",
    }
    assert module.layout_from_json(encoded) == layout


@pytest.mark.parametrize("field", ["device_dtype", "element_arrangement"])
def test_unknown_layout_enum_is_rejected(layouts, field):
    module, layout = layouts
    encoded = module.layout_to_json(layout)
    encoded[field] = "unknown"
    with pytest.raises(KeyError):
        module.layout_from_json(encoded)
