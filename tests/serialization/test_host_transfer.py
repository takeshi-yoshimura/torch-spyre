# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raw host-transfer primitives; no loader or scheduling policy."""

import pytest
import torch

DEVICE = torch.device("spyre")


def _default(t):
    return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=DEVICE)


def test_raw_copy_from_owned_host_buffer():
    import gc

    from torch_spyre._C import (
        allocate_host_buffer,
        convert_h2d_into,
        copy_tensor,
        dma_bytes,
        enqueue_copy_raw_h2d,
        synchronize_raw_h2d,
    )

    src = torch.randn(64, 64).half()
    dst = _default(src)
    reference = _default(src)
    copy_tensor(src, reference, non_blocking=False)
    buffer = allocate_host_buffer(dma_bytes(dst), DEVICE)
    # A numpy/memoryview export must retain the C++ allocation.
    view = memoryview(buffer.numpy())
    del buffer
    gc.collect()
    convert_h2d_into([(view, src, dst)], num_threads=1)
    buffer = torch.frombuffer(view, dtype=torch.uint8)
    enqueue_copy_raw_h2d(buffer, dst)
    synchronize_raw_h2d(DEVICE)
    torch.testing.assert_close(dst.cpu(), reference.cpu(), rtol=0, atol=0)


def test_raw_copy_rejects_unaligned_buffer():
    from torch_spyre._C import allocate_host_buffer, dma_bytes, enqueue_copy_raw_h2d

    dst = _default(torch.randn(64, 64).half())
    buffer = allocate_host_buffer(dma_bytes(dst) + 1, DEVICE)
    with pytest.raises(RuntimeError, match="IOMMU-aligned"):
        enqueue_copy_raw_h2d(buffer[1:], dst)


def test_raw_copy_rejects_insufficient_capacity():
    from torch_spyre._C import allocate_host_buffer, enqueue_copy_raw_h2d

    dst = _default(torch.randn(64, 64).half())
    buffer = allocate_host_buffer(1, DEVICE)
    with pytest.raises((ValueError, RuntimeError)):
        enqueue_copy_raw_h2d(buffer, dst)


def test_host_convert_batch_rejects_insufficient_capacity():
    from torch_spyre._C import host_convert_batch

    src = torch.randn(64, 64).half()
    dst = _default(src)
    tiny = torch.empty(1, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="destination buffer is too small"):
        host_convert_batch(
            [(memoryview(tiny.numpy()), memoryview(src.numpy()), (src, dst))],
            num_threads=1,
        )


@pytest.mark.parametrize("fp8", [False, True])
def test_raw_d2h_preserves_device_bytes(fp8):
    from torch_spyre._C import (
        allocate_host_buffer,
        dma_bytes,
        enqueue_copy_raw_d2h,
        enqueue_copy_raw_h2d,
        get_spyre_tensor_layout,
        spyre_empty_with_layout,
        synchronize_raw_transfers,
    )

    src = torch.randn(128, 128, dtype=torch.float16).to(DEVICE)
    if fp8:
        scale = torch.ones(1, dtype=torch.float16, device=DEVICE)
        src = torch.ops.spyre.quantize_weight_fp8_with_scale(src, scale)
    buffer = allocate_host_buffer(dma_bytes(src), DEVICE)
    enqueue_copy_raw_d2h(buffer, src)
    synchronize_raw_transfers(DEVICE)
    dst = spyre_empty_with_layout(
        src.size(), src.stride(), src.dtype, get_spyre_tensor_layout(src), device=DEVICE
    )
    enqueue_copy_raw_h2d(buffer, dst)
    synchronize_raw_transfers(DEVICE)
    actual = allocate_host_buffer(dma_bytes(dst), DEVICE)
    enqueue_copy_raw_d2h(actual, dst)
    synchronize_raw_transfers(DEVICE)
    torch.testing.assert_close(actual, buffer, rtol=0, atol=0)


def test_raw_d2h_rejects_bad_buffer():
    from torch_spyre._C import allocate_host_buffer, dma_bytes, enqueue_copy_raw_d2h

    src = torch.ones(64, 64, dtype=torch.float16, device=DEVICE)
    buffer = allocate_host_buffer(dma_bytes(src) + 1, DEVICE)
    with pytest.raises(RuntimeError, match="IOMMU-aligned"):
        enqueue_copy_raw_d2h(buffer[1:], src)
    with pytest.raises((ValueError, RuntimeError)):
        enqueue_copy_raw_d2h(buffer[:1], src)
