//------------------------------------------------------------------------------
// rmm_wrap.cpp: C-callable wrapper for CUDA memory management
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a lightweight replacement for the original RAPIDS RMM-based wrapper.
// It provides the same C API that SuiteSparse:GraphBLAS expects when built with
// CUDA, but uses CUDA runtime allocation APIs directly.

#include "rmm_wrap.h"
#include "rmm_wrap.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

struct rmm_wrap_device_context
{
    bool initialized = false ;
    RMM_MODE mode = rmm_wrap_managed ;
    cudaStream_t main_stream = nullptr ;
    std::vector<cudaStream_t> stream_pool ;
    std::size_t next_stream = 0 ;
    rmm_wrap_alloc_map allocs ;
} ;

static std::vector<rmm_wrap_device_context> rmm_wrap_contexts ;

static int rmm_wrap_ensure_contexts (void)
{
    int ndevices = 0 ;
    cudaError_t e = cudaGetDeviceCount (&ndevices) ;
    if (e != cudaSuccess || ndevices <= 0) return (-1) ;
    if ((int) rmm_wrap_contexts.size ( ) < ndevices)
    {
        rmm_wrap_contexts.resize ((size_t) ndevices) ;
    }
    return (0) ;
}

static void *rmm_wrap_malloc_impl (std::size_t size, RMM_MODE mode)
{
    if (size == 0) return nullptr ;

    void *p = nullptr ;
    cudaError_t e = cudaSuccess ;

    switch (mode)
    {
        case rmm_wrap_host:
            p = std::malloc (size) ;
            break ;

        case rmm_wrap_host_pinned:
            e = cudaMallocHost (&p, size) ;
            if (e != cudaSuccess) p = nullptr ;
            break ;

        case rmm_wrap_device:
            e = cudaMalloc (&p, size) ;
            if (e != cudaSuccess) p = nullptr ;
            break ;

        default:
        case rmm_wrap_managed:
            e = cudaMallocManaged (&p, size, cudaMemAttachGlobal) ;
            if (e != cudaSuccess) p = nullptr ;
            break ;
    }

    return p ;
}

static void rmm_wrap_free_impl (void *p, RMM_MODE mode)
{
    if (p == nullptr) return ;

    switch (mode)
    {
        case rmm_wrap_host:
            std::free (p) ;
            break ;

        case rmm_wrap_host_pinned:
            (void) cudaFreeHost (p) ;
            break ;

        case rmm_wrap_device:
        case rmm_wrap_managed:
        default:
            (void) cudaFree (p) ;
            break ;
    }
}

extern "C" {

bool rmm_wrap_is_initialized (void)
{
    for (auto &c : rmm_wrap_contexts)
    {
        if (c.initialized) return true ;
    }
    return false ;
}

void rmm_wrap_finalize (void)
{
    for (auto &c : rmm_wrap_contexts)
    {
        if (!c.initialized) continue ;
        for (auto s : c.stream_pool)
        {
            if (s != nullptr) (void) cudaStreamDestroy (s) ;
        }
        if (c.main_stream != nullptr) (void) cudaStreamDestroy (c.main_stream) ;
        c.stream_pool.clear ( ) ;
        c.allocs.clear ( ) ;
        c.next_stream = 0 ;
        c.main_stream = nullptr ;
        c.initialized = false ;
    }
    rmm_wrap_contexts.clear ( ) ;
}

int get_current_device (void)
{
    int device_id = 0 ;
    (void) cudaGetDevice (&device_id) ;
    return device_id ;
}

int rmm_wrap_initialize
(
    uint32_t device_id,
    RMM_MODE mode,
    size_t init_pool_size,
    size_t max_pool_size,
    size_t stream_pool_size
)
{
    (void) init_pool_size ;
    (void) max_pool_size ;

    if (rmm_wrap_ensure_contexts ( ) != 0) return (-1) ;
    if (device_id >= rmm_wrap_contexts.size ( )) return (-1) ;

    cudaError_t e = cudaSetDevice ((int) device_id) ;
    if (e != cudaSuccess) return (-1) ;

    auto &c = rmm_wrap_contexts [device_id] ;
    if (c.initialized) return (0) ;

    c.mode = mode ;

    e = cudaStreamCreateWithFlags (&c.main_stream, cudaStreamNonBlocking) ;
    if (e != cudaSuccess) return (-1) ;

    const std::size_t nstreams = (stream_pool_size == 0) ? 1 : stream_pool_size ;
    c.stream_pool.resize (nstreams, nullptr) ;
    for (std::size_t i = 0 ; i < nstreams ; i++)
    {
        e = cudaStreamCreateWithFlags (&c.stream_pool [i], cudaStreamNonBlocking) ;
        if (e != cudaSuccess) return (-1) ;
    }

    c.next_stream = 0 ;
    c.initialized = true ;
    return (0) ;
}

int rmm_wrap_initialize_all_same
(
    RMM_MODE mode,
    size_t init_pool_size,
    size_t max_pool_size,
    size_t stream_pool_size
)
{
    if (rmm_wrap_ensure_contexts ( ) != 0) return (-1) ;
    for (uint32_t device_id = 0 ; device_id < rmm_wrap_contexts.size ( ) ; device_id++)
    {
        if (rmm_wrap_initialize (device_id, mode, init_pool_size, max_pool_size,
            stream_pool_size) != 0)
        {
            return (-1) ;
        }
    }
    return (0) ;
}

void *rmm_wrap_malloc (size_t size)
{
    if (rmm_wrap_ensure_contexts ( ) != 0) return nullptr ;

    const int device_id = get_current_device ( ) ;
    if (device_id < 0 || device_id >= (int) rmm_wrap_contexts.size ( )) return nullptr ;

    auto &c = rmm_wrap_contexts [(size_t) device_id] ;
    if (!c.initialized)
    {
        if (rmm_wrap_initialize ((uint32_t) device_id, rmm_wrap_managed, 0, 0, 1) != 0)
        {
            return nullptr ;
        }
    }

    void *p = rmm_wrap_malloc_impl (size, c.mode) ;
    if (p != nullptr)
    {
        c.allocs [p] = rmm_wrap_allocation_info { size, (int) c.mode } ;
    }
    return p ;
}

void *rmm_wrap_calloc (size_t n, size_t size)
{
    size_t bytes = n * size ;
    void *p = rmm_wrap_malloc (bytes) ;
    if (p == nullptr) return nullptr ;

    const int device_id = get_current_device ( ) ;
    auto &c = rmm_wrap_contexts [(size_t) device_id] ;
    if (c.mode == rmm_wrap_host)
    {
        std::memset (p, 0, bytes) ;
    }
    else
    {
        (void) cudaMemsetAsync (p, 0, bytes, c.main_stream) ;
        (void) cudaStreamSynchronize (c.main_stream) ;
    }
    return p ;
}

void *rmm_wrap_realloc (void *p, size_t newsize)
{
    if (p == nullptr) return rmm_wrap_malloc (newsize) ;
    if (newsize == 0)
    {
        rmm_wrap_free (p) ;
        return nullptr ;
    }

    const int device_id = get_current_device ( ) ;
    if (device_id < 0 || device_id >= (int) rmm_wrap_contexts.size ( )) return nullptr ;
    auto &c = rmm_wrap_contexts [(size_t) device_id] ;

    auto it = c.allocs.find (p) ;
    if (it == c.allocs.end ( ))
    {
        void *q = rmm_wrap_malloc (newsize) ;
        if (q == nullptr) return nullptr ;
        if (c.mode == rmm_wrap_host) std::memcpy (q, p, newsize) ;
        else
        {
            (void) cudaMemcpyAsync (q, p, newsize, cudaMemcpyDefault, c.main_stream) ;
            (void) cudaStreamSynchronize (c.main_stream) ;
        }
        return q ;
    }

    const size_t oldsize = it->second.size ;
    const RMM_MODE oldmode = (RMM_MODE) it->second.mode ;

    void *q = rmm_wrap_malloc_impl (newsize, oldmode) ;
    if (q == nullptr) return nullptr ;

    const size_t ncopy = std::min (oldsize, newsize) ;
    if (oldmode == rmm_wrap_host)
    {
        std::memcpy (q, p, ncopy) ;
    }
    else
    {
        (void) cudaMemcpyAsync (q, p, ncopy, cudaMemcpyDefault, c.main_stream) ;
        (void) cudaStreamSynchronize (c.main_stream) ;
    }

    rmm_wrap_free_impl (p, oldmode) ;
    c.allocs.erase (it) ;
    c.allocs [q] = rmm_wrap_allocation_info { newsize, (int) oldmode } ;
    return q ;
}

void rmm_wrap_free (void *p)
{
    if (p == nullptr) return ;
    if (rmm_wrap_contexts.empty ( )) return ;

    const int device_id = get_current_device ( ) ;
    if (device_id < 0 || device_id >= (int) rmm_wrap_contexts.size ( )) return ;
    auto &c = rmm_wrap_contexts [(size_t) device_id] ;

    auto it = c.allocs.find (p) ;
    if (it == c.allocs.end ( ))
    {
        (void) cudaFree (p) ;
        return ;
    }

    const RMM_MODE mode = (RMM_MODE) it->second.mode ;
    rmm_wrap_free_impl (p, mode) ;
    c.allocs.erase (it) ;
}

void *rmm_wrap_allocate (size_t *size)
{
    if (size == nullptr) return nullptr ;
    return rmm_wrap_malloc (*size) ;
}

void rmm_wrap_deallocate (void *p, size_t size)
{
    (void) size ;
    rmm_wrap_free (p) ;
}

void *rmm_wrap_get_next_stream_from_pool (void)
{
    const int device_id = get_current_device ( ) ;
    if (device_id < 0 || device_id >= (int) rmm_wrap_contexts.size ( )) return nullptr ;
    auto &c = rmm_wrap_contexts [(size_t) device_id] ;
    if (!c.initialized) return nullptr ;
    if (c.stream_pool.empty ( )) return (void *) c.main_stream ;
    cudaStream_t s = c.stream_pool [c.next_stream % c.stream_pool.size ( )] ;
    c.next_stream++ ;
    return (void *) s ;
}

void *rmm_wrap_get_stream_from_pool (size_t stream_id)
{
    const int device_id = get_current_device ( ) ;
    if (device_id < 0 || device_id >= (int) rmm_wrap_contexts.size ( )) return nullptr ;
    auto &c = rmm_wrap_contexts [(size_t) device_id] ;
    if (!c.initialized) return nullptr ;
    if (c.stream_pool.empty ( )) return (void *) c.main_stream ;
    return (void *) c.stream_pool [stream_id % c.stream_pool.size ( )] ;
}

void *rmm_wrap_get_main_stream (void)
{
    const int device_id = get_current_device ( ) ;
    if (device_id < 0 || device_id >= (int) rmm_wrap_contexts.size ( )) return nullptr ;
    auto &c = rmm_wrap_contexts [(size_t) device_id] ;
    if (!c.initialized) return nullptr ;
    return (void *) c.main_stream ;
}

} // extern "C"

