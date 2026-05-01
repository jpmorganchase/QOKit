/**
 * Buffer management: huge pages, NUMA-aware first-touch.
 */

#include "alloc.h"
#include <cstring>

#ifdef __linux__
#include <sys/mman.h>
#endif


void hint_huge_pages(void* ptr, size_t bytes) {
#ifdef __linux__
    if (bytes >= 2 * 1024 * 1024) {
        // MADV_HUGEPAGE = 14 on Linux
        // This is a hint for Transparent Huge Pages (THP).
        // For guaranteed huge pages, use mmap with MAP_HUGETLB.
        madvise(ptr, bytes, 14);
    }
#else
    (void)ptr;
    (void)bytes;
#endif
}


std::vector<C128> alloc_huge(size_t n) {
    std::vector<C128> v(n);
    hint_huge_pages(v.data(), n * sizeof(C128));
    return v;
}


void numa_first_touch_zero(C128* data, size_t n) {
    // On NUMA systems, physical pages are allocated on the node of the
    // thread that first touches them. By zeroing from a parallel region
    // with static scheduling, each thread touches its own chunk, ensuring
    // pages are allocated on the local NUMA node.
    //
    // This matters for arrays > L3 cache size (typically > 40MB on a
    // 91-core system). Without first-touch, all pages land on the node
    // that did the allocation (often node 0), causing remote memory
    // accesses from all other sockets.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        data[i] = C128(0.0, 0.0);
    }
}
