// mirror_inherit::runs (gpu/vk/mirror_inherit.h): a new pool's mirror takes from a retired pool's mirror only the pages not
// written since that mirror's last sync. No game, no GPU.
#include "../gpu/vk/mirror_inherit.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

namespace {

constexpr uint64_t kShift = 12, kPage = 4096, kRegion = 512;   // (vk_memory.cpp's page and 2 MB region)

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) std::printf("FAIL: %s\n", what), failures++;
}

struct Case {
    std::map<uint64_t, uint32_t> gens;   // page -> generation (0 where absent)
    uint32_t synced = 10;
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> runs;
    uint64_t skipped = 0;
    void run(uint64_t a, uint64_t b, uint64_t r_lo, uint64_t p_lo) {
        runs.clear();
        skipped = mirror_inherit::runs(
            a, b, r_lo, p_lo, synced, kShift, kRegion,
            [&](uint64_t region) {
                for (uint64_t p = region * kRegion; p < (region + 1) * kRegion; p++)
                    if (gens.count(p) && gens[p] > synced) return false;
                return true;
            },
            [&](uint64_t page) { return gens.count(page) ? gens[page] : 0u; },
            [&](uint64_t src, uint64_t dst, uint64_t size) { runs.emplace_back(src, dst, size); });
    }
};

} // namespace

int main() {
    const uint64_t r_lo = 0x100000000ull, p_lo = 0x0ff800000ull;   // (the new pool starts 8 MB before the retired one)
    {   // nothing written since the sync: the whole part, one run
        Case c;
        c.run(r_lo, r_lo + (8 << 20), r_lo, p_lo);
        expect(c.skipped == 0, "a clean part leaves nothing to guest memory");
        expect(c.runs.size() == 1, "a clean part is one run");
        expect(c.runs.size() == 1 && c.runs[0] == std::make_tuple(uint64_t{0}, r_lo - p_lo, uint64_t{8 << 20}),
               "the run's offsets are the retired and the new mirror's");
    }
    {   // a page the CPU wrote since the sync is left to guest memory; the run around it splits
        Case c;
        const uint64_t page = (r_lo >> kShift) + 3;
        c.gens[page] = 11;
        c.run(r_lo, r_lo + 16 * kPage, r_lo, p_lo);
        expect(c.skipped == kPage, "the newer page is the bytes left to guest memory");
        expect(c.runs.size() == 2, "the part splits around the newer page");
        if (c.runs.size() == 2) {
            expect(std::get<2>(c.runs[0]) == 3 * kPage && std::get<0>(c.runs[0]) == 0, "before it: three pages");
            expect(std::get<0>(c.runs[1]) == 4 * kPage && std::get<2>(c.runs[1]) == 12 * kPage, "after it: the other twelve");
        }
    }
    {   // a page stamped at or before the sync (what the GPU wrote then) comes from the retired mirror
        Case c;
        c.gens[(r_lo >> kShift) + 5] = 10;
        c.gens[(r_lo >> kShift) + 6] = 2;
        c.run(r_lo, r_lo + 16 * kPage, r_lo, p_lo);
        expect(c.skipped == 0 && c.runs.size() == 1, "pages up to the sync's generation are inherited");
    }
    {   // a part that does not start or end on a page: the runs stay inside it
        Case c;
        c.gens[(r_lo >> kShift) + 1] = 12;
        const uint64_t a = r_lo + 100, b = r_lo + 3 * kPage - 100;
        c.run(a, b, r_lo, p_lo);
        uint64_t total = c.skipped;
        for (const auto& run : c.runs) total += std::get<2>(run);
        expect(total == b - a, "the runs and the skipped bytes cover the part exactly");
        expect(c.skipped == kPage, "the newer page whole");
        expect(!c.runs.empty() && std::get<0>(c.runs[0]) == 100, "the first run starts where the part does");
    }
    {   // a newer page in a later 2 MB region: the clean region before it is one step, and the runs still meet
        Case c;
        const uint64_t page = (r_lo >> kShift) + kRegion + 7;
        c.gens[page] = 20;
        c.run(r_lo, r_lo + 2 * kRegion * kPage, r_lo, p_lo);
        expect(c.skipped == kPage, "one page left");
        expect(c.runs.size() == 2, "one run before it (the clean region and the pages after), one after");
        if (c.runs.size() == 2) expect(std::get<2>(c.runs[0]) == (kRegion + 7) * kPage, "the first run spans the clean region");
    }
    if (failures) return EXIT_FAILURE;
    std::printf("PASS: a remade pool's mirror takes only the pages not written since the retired mirror's last sync\n");
    return EXIT_SUCCESS;
}
