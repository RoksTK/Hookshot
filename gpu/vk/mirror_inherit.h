// Which part of a retired pool's mirror a new pool's mirror takes over (vk_memory.cpp, mirror_make).
//
// A pool remade larger gets a new mirror: guest memory whole, then the retired pools' mirrors over their part, for what
// the GPU wrote there and only there. A page the CPU wrote after the retired mirror's last sync holds the game's newer
// data in guest memory, and the old mirror's copy of it is stale. Copied over the new mirror, it counted as synced and
// was never fixed: OoT writes its light probes when an area comes in, so after a pool was remade the fog read the old
// area's probes for good (white glow, black or coloured blocks over the fog, dark shadows; a save reload did not clear
// it, a restart did). Only the pages not stamped since the retired mirror's last sync are taken from it.
#pragma once

#include <algorithm>
#include <cstdint>

namespace mirror_inherit {

/// The runs of [a, b) (guest addresses, inside both pools) to copy from the retired mirror: pages whose generation is at
/// most synced, the retired mirror's last sync. region_clean(r): every page of 2 MB region r is at most synced (the
/// stamp map's summary); page_gen(p): page p's generation. emit(src, dst, size) gets each run, merged where contiguous,
/// with src an offset in the retired mirror (from r_lo) and dst one in the new mirror (from p_lo). Returns the bytes left
/// to guest memory.
template <class RegionClean, class PageGen, class Emit>
uint64_t runs(uint64_t a, uint64_t b, uint64_t r_lo, uint64_t p_lo, uint32_t synced, uint64_t page_shift, uint64_t region_pages,
              RegionClean region_clean, PageGen page_gen, Emit emit) {
    uint64_t skipped = 0, run_lo = 0, run_hi = 0;
    const auto flush = [&] {
        if (run_hi > run_lo) emit(run_lo - r_lo, run_lo - p_lo, run_hi - run_lo);
        run_lo = run_hi = 0;
    };
    const uint64_t page_size = uint64_t{1} << page_shift;
    for (uint64_t page = a >> page_shift; page < (b + page_size - 1) >> page_shift;) {
        const bool whole_region = page % region_pages == 0 && region_clean(page / region_pages);
        const uint64_t step = whole_region ? region_pages : 1;
        const uint64_t lo = std::max(page << page_shift, a), hi = std::min((page + step) << page_shift, b);
        if (hi > lo) {
            if (whole_region || page_gen(page) <= synced) {
                if (run_hi != lo) flush(), run_lo = lo;
                run_hi = hi;
            } else {
                flush();
                skipped += hi - lo;
            }
        }
        page += step;
    }
    flush();
    return skipped;
}

} // namespace mirror_inherit
