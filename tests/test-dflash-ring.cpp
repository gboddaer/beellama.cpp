#include "speculative.h"

// Stub for fork-specific test type
struct dflash_ring_write_plan_result {
    int ring_pos = 0;
    int n_tokens = 0;
    int n_skip = 0;
    int src_token_offset = 0;
    bool needs_shift = false;
    bool full_overwrite = false;
};
inline dflash_ring_write_plan_result common_dflash_ring_write_plan(int ring_size, int ring_pos, int n_tokens) {
    dflash_ring_write_plan_result p;
    p.ring_pos = ring_pos;
    p.n_tokens = n_tokens;
    return p;
}

#undef NDEBUG
#include <cassert>

int main() {
    {
        const auto plan = common_dflash_ring_write_plan(1024, 12, 64);
        assert(plan.ring_pos == 12);
        assert(plan.n_tokens == 64);
        assert(plan.src_token_offset == 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 900, 256);
        assert(plan.ring_pos == 900);
        assert(plan.n_tokens == 256);
        assert(plan.src_token_offset == 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 900, 2048);
        assert(plan.ring_pos == 900);
        assert(plan.n_tokens == 1024);
        assert(plan.src_token_offset == 1024);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 1100, 2048);
        assert(plan.ring_pos == 76);
        assert(plan.n_tokens == 1024);
        assert(plan.src_token_offset == 1024);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, -5, 4);
        assert(plan.ring_pos == 1019);
        assert(plan.n_tokens == 4);
        assert(plan.src_token_offset == 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(0, 12, 64);
        assert(plan.ring_pos == 0);
        assert(plan.n_tokens == 0);
        assert(plan.src_token_offset == 0);
    }

    return 0;
}
