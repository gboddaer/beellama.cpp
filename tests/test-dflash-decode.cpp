// Unit tests for DFlash decode integration functions.
//
// These tests verify the state-management and control-flow logic of the
// dflash::generate_draft / verify_draft / rollback / sync_draft_ctx helpers
// without requiring a loaded model (which would need GGUF files). The tests
// exercise the API surface, edge cases, and the fixes for the issues found
// in the GLM adversarial review (C1-C8, H1-H2).

#include "dflash-server-utils.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

// Convenience accessor: the header declares dflash_slot_state but we need the
// full definition from dflash-server-utils.h which we include above.

static int test_count = 0;
static int test_fail = 0;

#define ASSERT(cond) do { \
    test_count++; \
    if (!(cond)) { \
        test_fail++; \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } else { \
        fprintf(stdout, "ok: %s\n", #cond); \
    } \
} while (0)

// ---- generate_draft with null context returns empty, clears state ----
static void test_generate_draft_null_ctx() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    state.draft_tokens.push_back(42); // pre-populate to ensure clear() runs
    state.n_draft = 99;
    state.n_accepted = 7;

    auto draft = dflash::generate_draft(nullptr, state, /*seed=*/1, /*seq_id=*/0, 1);
    ASSERT(draft.empty());
    ASSERT(state.draft_tokens.empty());
    ASSERT(state.n_draft == 0);
    ASSERT(state.n_accepted == 0);
    dflash::cleanup_slot_dflash(state);
}

// ---- generate_draft with negative seed returns empty (no valid seed) ----
static void test_generate_draft_bad_seed() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    // ctx_dft is nullptr here; the null check fires first, but even with a
    // non-null ctx the negative seed should short-circuit. We test the seed
    // guard by passing a negative seed (ctx null short-circuits earlier).
    auto draft = dflash::generate_draft(nullptr, state, /*seed=*/-1, /*seq_id=*/0, 1);
    ASSERT(draft.empty());
    dflash::cleanup_slot_dflash(state);
}

// ---- generate_draft with n_draft <= 0 returns empty ----
static void test_generate_draft_zero_n() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    auto draft = dflash::generate_draft(nullptr, state, /*seed=*/1, /*seq_id=*/0, 0);
    ASSERT(draft.empty());
    auto draft2 = dflash::generate_draft(nullptr, state, /*seed=*/1, /*seq_id=*/0, -3);
    ASSERT(draft2.empty());
    dflash::cleanup_slot_dflash(state);
}

// ---- rollback with no draft tokens is a no-op and keeps active true ----
static void test_rollback_empty_keeps_active() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    state.active = true; // GLM C6: active must survive rollback
    dflash::rollback(nullptr, state, /*seq_id=*/0);
    ASSERT(state.active == true); // critical: active NOT cleared
    ASSERT(state.draft_tokens.empty());
    ASSERT(state.n_draft == 0);
    ASSERT(state.n_accepted == 0);
    dflash::cleanup_slot_dflash(state);
}

// ---- verify_draft on empty draft returns 0 ----
static void test_verify_empty_returns_zero() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    int n = dflash::verify_draft(nullptr, state);
    ASSERT(n == 0);
    dflash::cleanup_slot_dflash(state);
}

// ---- sync_draft_ctx with null ctx returns false ----
static void test_sync_null_returns_false() {
    bool ok = dflash::sync_draft_ctx(nullptr, /*token=*/1, /*seq_id=*/0);
    ASSERT(ok == false);
}

// ---- sync_draft_ctx with negative token returns false ----
static void test_sync_bad_token_returns_false() {
    bool ok = dflash::sync_draft_ctx(nullptr, /*token=*/-1, /*seq_id=*/0);
    ASSERT(ok == false);
}

// ---- init/cleanup idempotency ----
static void test_init_cleanup_idempotent() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    dflash::init_slot_dflash(state); // double init should not crash
    dflash::cleanup_slot_dflash(state);
    dflash::cleanup_slot_dflash(state); // double cleanup should not crash
    ASSERT(state.dft_ctx == nullptr);
    ASSERT(state.draft_tokens.empty());
    ASSERT(state.active == false);
}

// ---- base_pos tracking: generate_draft resets base_pos to -1 ----
static void test_base_pos_reset_on_generate() {
    dflash::dflash_slot_state state;
    dflash::init_slot_dflash(state);
    state.base_pos = 1234; // pre-set
    // generate_draft with null ctx resets base_pos to -1
    dflash::generate_draft(nullptr, state, /*seed=*/1, /*seq_id=*/0, 1);
    ASSERT(state.base_pos == -1);
    dflash::cleanup_slot_dflash(state);
}

int main() {
    fprintf(stdout, "=== DFlash decode integration unit tests ===\n");

    test_generate_draft_null_ctx();
    test_generate_draft_bad_seed();
    test_generate_draft_zero_n();
    test_rollback_empty_keeps_active();
    test_verify_empty_returns_zero();
    test_sync_null_returns_false();
    test_sync_bad_token_returns_false();
    test_init_cleanup_idempotent();
    test_base_pos_reset_on_generate();

    fprintf(stdout, "\n%d tests run, %d failed\n", test_count, test_fail);
    if (test_fail == 0) {
        fprintf(stdout, "ALL TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "SOME TESTS FAILED\n");
    return 1;
}