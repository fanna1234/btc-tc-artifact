//------------------------------------------------------------------------------
// rmm_wrap/rmm_wrap.hpp
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Minimal C++ helpers for rmm_wrap.cpp.
// This implementation intentionally avoids the RAPIDS RMM dependency to keep
// GraphBLAS CUDA builds self-contained and compatible with newer CUDA toolkits.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct rmm_wrap_allocation_info
{
    std::size_t size ;
    int mode ;
} ;

using rmm_wrap_alloc_map = std::unordered_map<void *, rmm_wrap_allocation_info> ;
