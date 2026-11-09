/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "host_transfer.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <flex/flex.hpp>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>

#include "module.h"
#include "spyre_composite_address.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {

size_t host_buffer_alignment() {
  auto* ctx = flex::RuntimeContext::getInstance();
  TORCH_CHECK(ctx != nullptr, "Spyre runtime not initialized");
  const auto type = ctx->getDeviceType();
  TORCH_CHECK(type == flex::DeviceType::PF || type == flex::DeviceType::VF,
              "Host transfers require an AF device");
  // PF and VF IOMMU mappers both align host mappings to the OS page size.
  const long page_size = sysconf(_SC_PAGESIZE);
  TORCH_CHECK(page_size > 0, "Unable to query host page size");
  return std::max(sizeof(void*), static_cast<size_t>(page_size));
}

at::Tensor allocate_host_buffer(size_t nbytes, c10::Device device) {
  TORCH_CHECK(device.is_privateuseone(),
              "allocate_host_buffer: device must be Spyre");
  TORCH_CHECK(nbytes > 0 && nbytes <= static_cast<size_t>(
                                        std::numeric_limits<int64_t>::max()),
              "allocate_host_buffer: invalid byte size ", nbytes);
  c10::DeviceGuard guard(device);
  auto* ctx = flex::RuntimeContext::getInstance();
  TORCH_CHECK(ctx != nullptr, "allocate_host_buffer: Spyre runtime not initialized");
  const size_t alignment = host_buffer_alignment();
  void* pointer = nullptr;
  const int error = posix_memalign(&pointer, alignment, nbytes);
  TORCH_CHECK(error == 0, "allocate_host_buffer: posix_memalign failed: ", error);
  auto storage = std::shared_ptr<void>(pointer, std::free);
  return at::from_blob(
      pointer, {static_cast<int64_t>(nbytes)}, [storage](void*) {},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
}

size_t dma_bytes(const at::Tensor& dst) {
  TORCH_CHECK(dst.is_privateuseone(),
              "dma_bytes: dst must be a Spyre tensor, got ", dst.device());
  const flex::CompositeAddress* addr = get_composite_address(dst);
  return addr->is_single_chunk() ? addr->total_size() : 0;
}

void convert_h2d_into(const std::vector<HostConvertTarget>& targets,
                     unsigned int num_threads) {
  if (targets.empty()) return;

  // Reserved up front: batch holds pointers into dcis.
  std::vector<DataConversionInfo> dcis;
  dcis.reserve(targets.size());
  std::vector<flex::HostConvertEntry> batch;
  batch.reserve(targets.size());

  // Validate everything before converting anything, so a bad entry leaves
  // every buffer untouched.
  for (const auto& t : targets) {
    TORCH_CHECK(t.cpu_tensor.device().is_cpu(),
                "convert_h2d_into: source must be on CPU, got ",
                t.cpu_tensor.device());
    TORCH_CHECK(t.dst_tensor.is_privateuseone(),
                "convert_h2d_into: dst_tensor must be a Spyre tensor, got "
                "device ", t.dst_tensor.device());
    const size_t bytes = dma_bytes(t.dst_tensor);
    TORCH_CHECK(bytes != 0 && bytes <= t.capacity,
                "convert_h2d_into: dst_tensor must be a single-chunk "
                "allocation of at most ", t.capacity, " B, got ", bytes, " B");

    // Same DCI as SpyreStream::copyAsync (the copy_tensor path).
    dcis.push_back(generate_dci(&t.cpu_tensor, &t.dst_tensor,
                                get_spyre_tensor_layout(t.dst_tensor),
                                /*host2device=*/true));
    batch.push_back(flex::HostConvertEntry{
        t.dst,
        t.cpu_tensor.storage().data(),
        &dcis.back(),
        t.capacity,
    });
  }

  if (num_threads == 0) {
    num_threads = std::max(1u, std::thread::hardware_concurrency());
  }
  flex::hostConvertBatch(batch, num_threads);
}

}  // namespace spyre
