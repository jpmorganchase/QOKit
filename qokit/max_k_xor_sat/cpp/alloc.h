#pragma once
/**
 * Buffer management: ping-pong buffers, huge pages, NUMA-aware allocation.
 *
 * On a 91-core multi-socket system with 1TB RAM:
 * - Huge pages (2MB) reduce TLB pressure from 16M+ entries to ~32K
 * - NUMA-aware first-touch ensures data is local to the accessing socket
 * - Ping-pong buffers avoid per-iteration GiB-scale allocations
 */

#include <complex>
#include <vector>
#include <cstddef>

using C128 = std::complex<double>;

// Forward declarations
void hint_huge_pages(void* ptr, size_t bytes);
void numa_first_touch_zero(C128* data, size_t n);

/// Ping-pong double buffer for avoiding per-iteration allocations.
struct PingPong {
    std::vector<C128> buf[2];
    int cur = 0;

    PingPong() = default;

    /// Allocate both buffers for n elements with huge page hints.
    void resize(size_t n) {
        buf[0].resize(n);
        buf[1].resize(n);
        cur = 0;
        hint_huge_pages(buf[0].data(), n * sizeof(C128));
        hint_huge_pages(buf[1].data(), n * sizeof(C128));
    }

    C128* in()       { return buf[cur].data(); }
    C128* out()      { return buf[1 - cur].data(); }
    const C128* in() const  { return buf[cur].data(); }
    const C128* out() const { return buf[1 - cur].data(); }

    void flip() { cur = 1 - cur; }

    size_t size() const { return buf[0].size(); }
};

/// Templated ping-pong buffer for generic Scalar types.
template<typename Scalar>
struct PingPong_T {
    std::vector<Scalar> buf[2];
    int cur = 0;

    Scalar* in()  { return buf[cur].data(); }
    Scalar* out() { return buf[1 - cur].data(); }
    void flip()   { cur = 1 - cur; }
};

/// Allocate a vector with huge page hint (Linux THP / mmap fallback).
std::vector<C128> alloc_huge(size_t n);
