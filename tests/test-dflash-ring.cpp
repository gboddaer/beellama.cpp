#include "speculative.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>

static void assert_ring_split(int ring_size, int ring_pos, int n_tokens, int first_expected, int second_expected) {
    const auto plan = common_dflash_ring_write_plan(ring_size, ring_pos, n_tokens);
    if (plan.n_tokens == 0) {
        assert(first_expected == 0);
        assert(second_expected == 0);
        return;
    }

    const int first = std::min(plan.n_tokens, ring_size - plan.ring_pos);
    const int second = plan.n_tokens - first;
    assert(first == first_expected);
    assert(second == second_expected);
    assert(first >= 0);
    assert(second >= 0);
    assert(first + second == plan.n_tokens);
    assert(plan.ring_pos >= 0);
    assert(plan.ring_pos < ring_size);
}

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
        assert_ring_split(1024, 900, 256, 124, 132);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 900, 2048);
        assert(plan.ring_pos == 900);
        assert(plan.n_tokens == 1024);
        assert(plan.src_token_offset == 1024);
        assert_ring_split(1024, 900, 2048, 124, 900);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 1100, 2048);
        assert(plan.ring_pos == 76);
        assert(plan.n_tokens == 1024);
        assert(plan.src_token_offset == 1024);
        assert_ring_split(1024, 1100, 2048, 948, 76);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, -5, 4);
        assert(plan.ring_pos == 1019);
        assert(plan.n_tokens == 4);
        assert(plan.src_token_offset == 0);
        assert_ring_split(1024, -5, 4, 4, 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(0, 12, 64);
        assert(plan.ring_pos == 0);
        assert(plan.n_tokens == 0);
        assert(plan.src_token_offset == 0);
        assert_ring_split(0, 12, 64, 0, 0);
    }

    return 0;
}
