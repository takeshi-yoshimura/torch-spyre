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

#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <vector>

namespace spyre {

size_t host_buffer_alignment();

/** Allocate ordinary CPU storage aligned to the device's IOMMU requirement. */
at::Tensor allocate_host_buffer(size_t nbytes, c10::Device device);

/** Physical transfer size, or 0 for unsupported multi-chunk allocations. */
size_t dma_bytes(const at::Tensor& dst);

struct HostConvertTarget {
  void* dst;
  size_t capacity;
  at::Tensor cpu_tensor;
  at::Tensor dst_tensor;
};

/** Convert host tensors directly into caller-owned device-format buffers. */
void convert_h2d_into(const std::vector<HostConvertTarget>& targets,
                     unsigned int num_threads);

}  // namespace spyre
