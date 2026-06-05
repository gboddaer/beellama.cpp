#include "../tools/server/server-context.h"
#include "../common/common.h"

#undef NDEBUG
#include <cassert>

int main() {
    common_params params;
    params.n_batch  = 1;
    params.n_ubatch = 1;
    params.speculative.set_type(COMMON_SPECULATIVE_TYPE_DFLASH);
    params.speculative.draft.n_max = 12;
    params.speculative.n_max = 12;
    params.speculative.branch_budget = 0;

    llama_context_params cparams = common_context_params_to_llama(params);
    assert(cparams.n_rs_seq == 0);
    assert(cparams.n_batch == 1);
    assert(cparams.n_ubatch == 1);

    params.speculative.branch_budget = 4;
    cparams = common_context_params_to_llama(params);
    assert(cparams.n_rs_seq == 0);

    params.speculative.set_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    params.speculative.branch_budget = 0;
    cparams = common_context_params_to_llama(params);
    assert(cparams.n_rs_seq == 12);
    assert(cparams.n_batch >= 13);
    assert(cparams.n_ubatch >= 13);

    params.speculative.set_type(COMMON_SPECULATIVE_TYPE_DFLASH);
    params.speculative.branch_budget = 0;
    server_dflash_recurrent_rollback_plan plan =
        server_context_dflash_recurrent_rollback_plan(
                params.speculative,
                /*target_recurrent_or_hybrid =*/ true);
    assert(plan.needs_backup_sequences);
    assert(!plan.needs_attention_backup_streams);

    plan = server_context_dflash_recurrent_rollback_plan(
            params.speculative,
            /*target_recurrent_or_hybrid =*/ true);
    assert(plan.needs_backup_sequences);
    assert(!plan.needs_attention_backup_streams);

    params.speculative.branch_budget = 4;
    plan = server_context_dflash_recurrent_rollback_plan(
            params.speculative,
            /*target_recurrent_or_hybrid =*/ true);
    assert(plan.needs_backup_sequences);
    assert(plan.needs_attention_backup_streams);

    plan = server_context_dflash_recurrent_rollback_plan(
            params.speculative,
            /*target_recurrent_or_hybrid =*/ false);
    assert(!plan.needs_backup_sequences);
    assert(!plan.needs_attention_backup_streams);

    return 0;
}
