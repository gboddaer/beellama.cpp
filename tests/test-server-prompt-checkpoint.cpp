#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>

int main() {
    server_prompt_checkpoint ckpt {
        /*.pos_min  = */ 1,
        /*.pos_max  = */ 2,
        /*.n_tokens = */ 3,
        /*.data     = */ std::vector<uint8_t>(128),
    };
    ckpt.ring_data.resize(256);

    ckpt.clear();

    assert(ckpt.pos_min == 0);
    assert(ckpt.pos_max == 0);
    assert(ckpt.n_tokens == 0);
    assert(ckpt.data.empty());
    assert(ckpt.data.capacity() == 0);
    assert(ckpt.ring_data.empty());
    assert(ckpt.ring_data.capacity() == 0);

    server_prompt prompt;
    for (int i = 0; i < 3; ++i) {
        auto & cur = prompt.checkpoints.emplace_back();
        cur.n_tokens = i + 1;
        cur.data_tgt.resize((size_t) (i + 1) * 100);
    }

    {
        server_prompt budgeted = server_prompt_clone_with_checkpoint_budget(prompt, 250, 600);
        assert(budgeted.checkpoints.size() == 1);
        assert(budgeted.checkpoints.front().n_tokens == 3);
        assert(server_prompt_checkpoints_size(budgeted.checkpoints) == 300);
    }

    {
        server_prompt budgeted = server_prompt_clone_with_checkpoint_budget(prompt, 600, 600);
        assert(budgeted.checkpoints.empty());
    }

    {
        server_prompt budgeted = server_prompt_clone_with_checkpoint_budget(prompt, 0, 0);
        assert(budgeted.checkpoints.size() == 3);
        assert(server_prompt_checkpoints_size(budgeted.checkpoints) == 600);
    }

    {
        common_prompt_checkpoint ckpt;
        ckpt.update_pos(17851, 17850, 17850);

        assert(server_prompt_checkpoint_matches_restore_window(
                ckpt,
                /*pos_min_thold =*/ 17851,
                /*pos_next =*/ 17851,
                /*is_recurrent_or_hybrid =*/ false));
        assert(!server_prompt_checkpoint_matches_restore_window(
                ckpt,
                /*pos_min_thold =*/ 17850,
                /*pos_next =*/ 17851,
                /*is_recurrent_or_hybrid =*/ false));

        ckpt.update_pos(17851, 0, 17850);
        assert(server_prompt_checkpoint_matches_restore_window(
                ckpt,
                /*pos_min_thold =*/ 0,
                /*pos_next =*/ 17851,
                /*is_recurrent_or_hybrid =*/ false));

        ckpt.update_pos(17851, 17850, 17850);
        assert(server_prompt_checkpoint_matches_restore_window(
                ckpt,
                /*pos_min_thold =*/ 0,
                /*pos_next =*/ 17850,
                /*is_recurrent_or_hybrid =*/ true));
        assert(!server_prompt_checkpoint_matches_restore_window(
                ckpt,
                /*pos_min_thold =*/ 0,
                /*pos_next =*/ 17849,
                /*is_recurrent_or_hybrid =*/ true));
    }

    {
        assert(server_prompt_effective_checkpoint_limit(
                /*configured_checkpoints =*/ 0,
                /*prompt_cache_boundary_required =*/ false) == 0);
        assert(server_prompt_effective_checkpoint_limit(
                /*configured_checkpoints =*/ 0,
                /*prompt_cache_boundary_required =*/ true) == 1);
        assert(server_prompt_effective_checkpoint_limit(
                /*configured_checkpoints =*/ 32,
                /*prompt_cache_boundary_required =*/ true) == 32);

        assert(server_prompt_checkpoint_creation_allowed(
                /*boundary_only =*/ true,
                /*n_before_user_known =*/ true,
                /*is_on_user =*/ true,
                /*is_after_user =*/ false,
                /*near_prompt_end =*/ false));
        assert(!server_prompt_checkpoint_creation_allowed(
                /*boundary_only =*/ true,
                /*n_before_user_known =*/ true,
                /*is_on_user =*/ false,
                /*is_after_user =*/ true,
                /*near_prompt_end =*/ true));
        assert(server_prompt_checkpoint_creation_allowed(
                /*boundary_only =*/ false,
                /*n_before_user_known =*/ true,
                /*is_on_user =*/ false,
                /*is_after_user =*/ true,
                /*near_prompt_end =*/ true));
    }

    prompt.data.main.resize(64);
    prompt.data.drft.resize(32);
    prompt.clear();
    assert(prompt.n_tokens() == 0);
    assert(prompt.data.size() == 0);
    assert(prompt.checkpoints.empty());

    return 0;
}
