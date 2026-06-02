#pragma once

#include "common.h"
#include "log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

static constexpr int SERVER_ADAPTIVE_DM_PROFIT_POSITIONS  = 128;
static constexpr int SERVER_ADAPTIVE_DM_PROFIT_DEPTHS     = SERVER_ADAPTIVE_DM_PROFIT_POSITIONS + 1;
static constexpr int SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES = SERVER_ADAPTIVE_DM_PROFIT_DEPTHS + 1;

static inline int server_adaptive_dm_probe_n_max(int base_n_max, float probe_fraction) {
    if (base_n_max <= 0) {
        return 0;
    }

    const int probe_n_max = std::max(2, (int) (base_n_max * probe_fraction));
    return std::min(probe_n_max, base_n_max);
}

static inline float server_adaptive_dm_required_fringe_for_n_max(
        int target_n_max,
        int base_n_max,
        float fringe_min,
        float fringe_max) {
    if (base_n_max <= 2 || target_n_max <= 2 || fringe_max <= fringe_min) {
        return fringe_min;
    }

    const int bounded_n_max = std::clamp(target_n_max, 2, base_n_max);
    const float t = (float)(bounded_n_max - 2) / (float)(base_n_max - 2);
    return std::clamp(fringe_min + (fringe_max - fringe_min) * t, fringe_min, fringe_max);
}

static inline bool server_adaptive_dm_should_preserve_for_continuation(float prompt_similarity, float keep_fraction) {
    return prompt_similarity > 0.90f && keep_fraction > 0.90f;
}

static inline bool server_adaptive_dm_uses_fringe_controller(common_speculative_dm_controller controller) {
    return controller == COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE;
}

static inline bool server_adaptive_dm_uses_profit_controller(common_speculative_dm_controller controller) {
    return controller == COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
}

static inline const char * server_adaptive_dm_controller_name(common_speculative_dm_controller controller) {
    switch (controller) {
        case COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE: return "fringe";
        case COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT: return "profit";
    }
    return "unknown";
}

// Initialize EWMA with first sample, then run normal update.
// Call this instead of checking dst == 0.0f (which is a valid EWMA value).
static inline void server_adaptive_dm_ewma_init_or_update(float & dst, float sample, float alpha, int32_t samples_seen) {
    if (!std::isfinite(sample)) {
        return;
    }
    alpha = std::clamp(alpha, 0.01f, 1.0f);
    if (samples_seen == 0) {
        dst = sample;
    } else {
        dst += alpha * (sample - dst);
    }
}

static inline int server_adaptive_dm_build_candidates(int base_n_max, int * out, int out_cap) {
    if (out_cap <= 0) {
        return 0;
    }

    int n = 0;
    auto add_unique = [&](int candidate) {
        if (candidate < 0 || candidate > base_n_max || n >= out_cap) {
            return;
        }
        bool exists = false;
        for (int i = 0; i < n; ++i) {
            if (out[i] == candidate) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            out[n++] = candidate;
        }
    };

    add_unique(0);
    if (base_n_max <= 11) {
        for (int candidate = 1; candidate <= base_n_max; ++candidate) {
            add_unique(candidate);
        }
    } else {
        const int ladder[] = {4, 8, 10, 12, 14, base_n_max};
        for (const int candidate : ladder) {
            add_unique(candidate);
        }
    }
    std::sort(out, out + n);
    return n;
}

static inline int server_adaptive_dm_next_explore_depth(int current_n, int base_n_max, float probe_fraction) {
    if (base_n_max <= 0) {
        return 0;
    }
    if (current_n <= 0) {
        return server_adaptive_dm_probe_n_max(base_n_max, probe_fraction);
    }

    int candidates[SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES];
    const int n_candidates = server_adaptive_dm_build_candidates(
            base_n_max, candidates, SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES);
    for (int i = 0; i < n_candidates; ++i) {
        if (candidates[i] > current_n) {
            return candidates[i];
        }
    }
    return std::min(current_n, base_n_max);
}

static inline float server_adaptive_dm_survival_expected_accept(
        const float * pos_accept_ewma,
        const int32_t * pos_samples,
        int n_positions,
        int depth,
        int min_samples,
        bool * ready) {
    bool is_ready = true;
    float expected = 0.0f;

    if (depth <= 0) {
        if (ready) {
            *ready = true;
        }
        return 0.0f;
    }

    // pos_accept_ewma[pos] records cumulative survival P(accepted at least pos+1).
    // Expected accepted tokens = sum of survival probabilities over positions.
    for (int pos = 0; pos < depth; ++pos) {
        if (pos >= n_positions || pos_samples[pos] < min_samples) {
            is_ready = false;
            break;
        }
        if (!std::isfinite(pos_accept_ewma[pos])) {
            is_ready = false;
            break;
        }
        expected += std::clamp(pos_accept_ewma[pos], 0.0f, 1.0f);
    }

    if (ready) {
        *ready = is_ready;
    }
    return is_ready ? expected : 0.0f;
}

static inline float server_adaptive_dm_score(float expected_output_tokens, float cycle_ms) {
    if (cycle_ms <= 0.0f || expected_output_tokens <= 0.0f
            || !std::isfinite(cycle_ms) || !std::isfinite(expected_output_tokens)) {
        return 0.0f;
    }
    return expected_output_tokens * 1000.0f / cycle_ms;
}

struct server_adaptive_dm_profit_candidate {
    int n = 0;
    float score = 0.0f;
    bool ready = false;
    bool estimated = false;
};

static inline server_adaptive_dm_profit_candidate server_adaptive_dm_best_profit_candidate(
        const int * candidates,
        int n_candidates,
        const bool * ready,
        const bool * estimated,
        const float * expected_accept,
        const float * cycle_ms) {
    server_adaptive_dm_profit_candidate best;
    for (int i = 0; i < n_candidates; ++i) {
        if (!ready[i]) {
            continue;
        }
        const float score = server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
        if (!best.ready || score > best.score) {
            best.ready = true;
            best.n = candidates[i];
            best.score = score;
            best.estimated = estimated[i];
        }
    }
    return best;
}

static inline int server_adaptive_dm_apply_profit_hysteresis(
        int current_n,
        int best_n,
        float current_score,
        float best_score,
        float raise_margin,
        float lower_margin,
        const int * candidates,
        int n_candidates) {
    if (best_n > current_n) {
        if (current_score > 0.0f && best_score < current_score * (1.0f + raise_margin)) {
            return current_n;
        }
        int next = current_n;
        for (int step = 0; step < 2 && next < best_n; step++) {
            int found = -1;
            for (int i = 0; i < n_candidates; i++) {
                if (candidates[i] > next) {
                    found = candidates[i];
                    break;
                }
            }
            if (found < 0) break;
            next = found;
        }
        return next;
    }
    if (best_n < current_n) {
        if (current_score > 0.0f && best_score < current_score * (1.0f + lower_margin)) {
            return current_n;
        }
        // Gradual downshift: move down one step on the ladder, not all the way to best_n.
        // Prevents collapsing from depth 16 to depth 5 on a few noisy samples.
        int next = current_n;
        for (int i = n_candidates - 1; i >= 0; --i) {
            if (candidates[i] < next) {
                return candidates[i];
            }
        }
        return current_n;
    }
    return current_n;
}

struct server_adaptive_dm_state {
    static constexpr int FRINGE_WINDOW = 32;
    static constexpr int FRINGE_REACH_POSITIONS = SERVER_ADAPTIVE_DM_PROFIT_POSITIONS;
    static constexpr int PROFIT_POSITIONS = SERVER_ADAPTIVE_DM_PROFIT_POSITIONS;
    static constexpr int PROFIT_DEPTHS = SERVER_ADAPTIVE_DM_PROFIT_DEPTHS;
    static constexpr int PROFIT_CANDIDATES = SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES;

    struct fringe_entry {
        int16_t  n_accepted;
        int16_t  n_draft;
        uint32_t epoch;
    };

    fringe_entry fringe_ring[FRINGE_WINDOW] = {};
    int   fringe_ring_idx   = 0;
    int   fringe_ring_count = 0;
    float rolling_fringe    = 0.0f;
    int32_t adaptive_n_max  = -1;
    int32_t adaptive_probe_counter = 0;
    int32_t off_dwell       = 0;
    int32_t explore_counter = 0;
    uint32_t fringe_epoch   = 0;
    int32_t fringe_epoch_reached[FRINGE_REACH_POSITIONS] = {};
    int32_t fringe_epoch_accepted[FRINGE_REACH_POSITIONS] = {};

    bool    dm_adaptive       = true;
    float   dm_fringe_min     = 0.30f;
    float   dm_fringe_max     = 0.50f;
    int32_t dm_off_dwell      = 8;
    int32_t dm_explore_interval = 12;
    int32_t dm_min_reach      = 3;
    int32_t dm_probe_interval = 16;
    float   dm_probe_fraction = 0.25f;
    int32_t dm_fringe_window  = 3;
    common_speculative_dm_controller dm_controller = COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
    float   dm_profit_min          = 0.05f;
    float   dm_profit_raise_margin = 0.05f;
    float   dm_profit_lower_margin = 0.05f;
    float   dm_profit_ewma_alpha   = 0.15f;
    int32_t dm_profit_min_samples  = 3;
    int32_t dm_profit_warmup       = 0;
    int32_t dm_profit_baseline_interval = 1024;

    struct profit_depth_stats {
        int32_t samples = 0;
        float draft_ms  = 0.0f;
        float verify_ms = 0.0f;
        float accept_ms = 0.0f;
        float cycle_ms  = 0.0f;
        float output_tokens = 0.0f;
        float actual_draft_tokens = 0.0f;
        float best_score = 0.0f;
        double total_output_tokens = 0.0;
        double total_cycle_ms = 0.0;
    };

    struct profit_config_key {
        int32_t base_n_max       = -1;
        int32_t branch_budget    = -1;
        int32_t draft_topk       = -1;
        int32_t dflash_cross_ctx = -1;
        float   draft_temp       = -1.0f;
        float   p_min            = -1.0f;
        int32_t target_top_k     = -1;
        int32_t target_has_grammar = -1;
        int32_t target_grammar_lazy = -1;
        float   target_temp      = -1.0f;
        float   target_top_p     = -1.0f;
        float   target_min_p     = -1.0f;
    };

    float   profit_pos_accept_ewma[PROFIT_POSITIONS] = {};
    int32_t profit_pos_samples[PROFIT_POSITIONS] = {};
    profit_depth_stats profit_depth[PROFIT_DEPTHS] = {};
    profit_depth_stats profit_baseline;
    profit_config_key profit_key;
    bool    profit_has_key = false;
    bool    profit_pending = false;
    int32_t profit_pending_requested_n_max = 0;
    int32_t profit_pending_n_draft = 0;
    int32_t profit_pending_n_accepted = 0;
    bool    profit_pending_tree = false;
    uint32_t profit_epoch = 0;
    float   profit_current_score = 0.0f;
    int32_t profit_last_recommended_n = -1;
    int32_t profit_consecutive_below_profit = 0;
    int32_t profit_cycles_since_baseline = 0;
    bool    profit_baseline_probe_pending = false;
    int32_t profit_baseline_probe_resume_n = -1;
    int32_t profit_off_probe_failures = 0;
    int32_t profit_off_probe_next_n = -1;
    int32_t profit_active_episode_n = -1;
    int32_t profit_active_episode_start_samples = 0;
    double  profit_active_episode_start_output_tokens = 0.0;
    double  profit_active_episode_start_cycle_ms = 0.0;

    struct fringe_decision {
        float fringe = 0.0f;
        int   reached = 0;
        int   accepted = 0;
    };

    void reset_fringe_epoch_reached() {
        std::fill_n(fringe_epoch_reached, FRINGE_REACH_POSITIONS, 0);
        std::fill_n(fringe_epoch_accepted, FRINGE_REACH_POSITIONS, 0);
    }

    void reset_request_state(bool preserve_active_recommendation = false) {
        const int32_t previous_recommended_n = profit_last_recommended_n;
        const int32_t resume_n_max =
            preserve_active_recommendation && previous_recommended_n > 0
                ? previous_recommended_n
                : -1;
        const bool can_resume =
            resume_n_max > 0 &&
            profit_has_key &&
            (profit_key.base_n_max <= 0 || resume_n_max <= profit_key.base_n_max);
        const int32_t warm_start_n_max =
            !preserve_active_recommendation && previous_recommended_n > 0
                ? previous_recommended_n
                : -1;

        std::fill_n(fringe_ring, FRINGE_WINDOW, fringe_entry{});
        fringe_ring_idx = 0;
        fringe_ring_count = 0;
        rolling_fringe = 0.0f;
        adaptive_n_max = -1;
        adaptive_probe_counter = 0;
        off_dwell = 0;
        explore_counter = 0;
        fringe_epoch++;
        reset_fringe_epoch_reached();
        reset_request_profit_state(false);
        if (can_resume) {
            adaptive_n_max = resume_n_max;
            profit_last_recommended_n = resume_n_max;
            profit_current_score = profit_score_for_depth(resume_n_max);
            profit_begin_active_episode(resume_n_max);
        } else if (warm_start_n_max > 0 &&
                profit_has_key &&
                (profit_key.base_n_max <= 0 || warm_start_n_max <= profit_key.base_n_max) &&
                profit_baseline_ready() &&
                profit_score_for_depth(warm_start_n_max) >= profit_score_for_depth(0) * (1.0f + dm_profit_min)) {
            adaptive_n_max = warm_start_n_max;
            profit_last_recommended_n = warm_start_n_max;
            profit_current_score = profit_score_for_depth(warm_start_n_max);
            profit_begin_active_episode(warm_start_n_max);
        }
    }

    void reset_request_profit_state(bool preserve_recommendation = false) {
        profit_pending = false;
        profit_pending_requested_n_max = 0;
        profit_pending_n_draft = 0;
        profit_pending_n_accepted = 0;
        profit_pending_tree = false;
        profit_consecutive_below_profit = 0;
        profit_current_score = 0.0f;
        if (!preserve_recommendation) {
            profit_last_recommended_n = -1;
        }
        profit_cycles_since_baseline = 0;
        profit_baseline_probe_pending = false;
        profit_baseline_probe_resume_n = -1;
        profit_off_probe_failures = 0;
        profit_off_probe_next_n = -1;
        profit_active_episode_n = -1;
        profit_active_episode_start_samples = 0;
        profit_active_episode_start_output_tokens = 0.0;
        profit_active_episode_start_cycle_ms = 0.0;
    }

    void reset_profit_state() {
        std::fill_n(profit_pos_accept_ewma, PROFIT_POSITIONS, 0.0f);
        std::fill_n(profit_pos_samples, PROFIT_POSITIONS, 0);
        std::fill_n(profit_depth, PROFIT_DEPTHS, profit_depth_stats{});
        profit_baseline = profit_depth_stats{};
        profit_key = profit_config_key{};
        profit_has_key = false;
        profit_epoch++;
        reset_request_profit_state();
        adaptive_n_max = -1;
        adaptive_probe_counter = 0;
        explore_counter = 0;
        off_dwell = 0;
    }

    void reset_profit_if_config_changed(
            const common_params_speculative & spec,
            int base_n_max,
            int32_t n_past,
            const common_params_sampling * sampling = nullptr) {
        (void) n_past;
        const profit_config_key next {
            base_n_max,
            spec.branch_budget,
            spec.draft_topk,
            spec.dflash_cross_ctx,
            spec.sample_temp,
            spec.p_min,
            sampling ? sampling->top_k : -1,
            sampling ? (common_grammar_value(sampling->grammar).empty() ? 0 : 1) : -1,
            sampling ? (sampling->grammar_lazy ? 1 : 0) : -1,
            sampling ? sampling->temp : -1.0f,
            sampling ? sampling->top_p : -1.0f,
            sampling ? sampling->min_p : -1.0f,
        };
        const bool changed =
            !profit_has_key ||
            profit_key.base_n_max       != next.base_n_max ||
            profit_key.branch_budget    != next.branch_budget ||
            profit_key.draft_topk       != next.draft_topk ||
            profit_key.dflash_cross_ctx != next.dflash_cross_ctx ||
            profit_key.draft_temp       != next.draft_temp ||
            profit_key.p_min            != next.p_min ||
            profit_key.target_top_k     != next.target_top_k ||
            profit_key.target_has_grammar != next.target_has_grammar ||
            profit_key.target_grammar_lazy != next.target_grammar_lazy ||
            profit_key.target_temp      != next.target_temp ||
            profit_key.target_top_p     != next.target_top_p ||
            profit_key.target_min_p     != next.target_min_p;
        if (!changed) {
            return;
        }

        reset_profit_state();
        profit_key = next;
        profit_has_key = true;
    }

    bool profit_baseline_ready() const {
        return profit_baseline.samples >= dm_profit_min_samples && profit_baseline.cycle_ms > 0.0f;
    }

    bool profit_depth_ready(int depth) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= dm_profit_min_samples &&
            profit_depth[depth].cycle_ms > 0.0f &&
            profit_depth[depth].output_tokens > 0.0f;
    }

    int profit_initial_probe_min_samples() const {
        return dm_profit_warmup > 0 ?
            std::max<int>(dm_profit_min_samples, dm_profit_warmup) :
            dm_profit_min_samples;
    }

    bool profit_initial_probe_depth_ready(int depth) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= profit_initial_probe_min_samples() &&
            profit_depth[depth].cycle_ms > 0.0f &&
            profit_depth[depth].output_tokens > 0.0f;
    }

    bool profit_has_ready_positive_depth() const {
        for (int d = 1; d < PROFIT_DEPTHS; ++d) {
            if (profit_depth_ready(d)) {
                return true;
            }
        }
        return false;
    }

    float profit_score_for_stats(const profit_depth_stats & stats) const {
        if (stats.samples < dm_profit_min_samples ||
                stats.total_cycle_ms <= 0.0 ||
                stats.total_output_tokens <= 0.0 ||
                !std::isfinite(stats.total_cycle_ms) ||
                !std::isfinite(stats.total_output_tokens)) {
            return 0.0f;
        }
        return server_adaptive_dm_score(
                (float) stats.total_output_tokens,
                (float) stats.total_cycle_ms);
    }

    float profit_score_for_depth(int depth) const {
        if (depth == 0) {
            if (!profit_baseline_ready()) {
                return 0.0f;
            }
            return profit_score_for_stats(profit_baseline);
        }
        if (!profit_depth_ready(depth)) {
            return 0.0f;
        }
        return profit_score_for_stats(profit_depth[depth]);
    }

    int profit_initial_probe_depths(int base_n_max, int * out, int out_cap) const {
        if (out_cap <= 0 || base_n_max <= 0) {
            return 0;
        }

        int n = 0;
        auto add_unique = [&](int value) {
            if (value <= 0 || value > base_n_max || n >= out_cap) {
                return;
            }
            for (int i = 0; i < n; ++i) {
                if (out[i] == value) {
                    return;
                }
            }
            out[n++] = value;
        };

        if (base_n_max >= 12) {
            add_unique(std::max(4, server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction)));
            add_unique(std::min(8, base_n_max));
            add_unique(std::min(10, base_n_max));
            add_unique(std::min(12, base_n_max));
        } else {
            const int shallow = server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction);
            const int mid = std::max(1, base_n_max / 2);
            add_unique(shallow);
            add_unique(mid);
            add_unique(base_n_max);
        }
        return n;
    }

    int profit_next_unready_probe_depth(int base_n_max) const {
        int probes[8];
        const int n_probes = profit_initial_probe_depths(base_n_max, probes, 8);
        for (int i = 0; i < n_probes; ++i) {
            if (!profit_initial_probe_depth_ready(probes[i])) {
                return probes[i];
            }
        }
        return -1;
    }

    int profit_first_initial_probe_depth(int base_n_max) const {
        int probes[8];
        const int n_probes = profit_initial_probe_depths(base_n_max, probes, 8);
        return n_probes > 0 ? probes[0] : 0;
    }

    int profit_next_initial_probe_depth_after(int current_n, int base_n_max, bool * wrapped = nullptr) const {
        int probes[8];
        const int n_probes = profit_initial_probe_depths(base_n_max, probes, 8);
        if (n_probes <= 0) {
            if (wrapped) {
                *wrapped = false;
            }
            return 0;
        }
        for (int i = 0; i < n_probes; ++i) {
            if (probes[i] > current_n) {
                if (wrapped) {
                    *wrapped = false;
                }
                return probes[i];
            }
        }
        if (wrapped) {
            *wrapped = true;
        }
        return probes[0];
    }

    int profit_off_probe_depths(int base_n_max, int * out, int out_cap) const {
        if (out_cap <= 0 || base_n_max <= 0) {
            return 0;
        }

        int n = 0;
        auto add_unique = [&](int value) {
            if (value <= 0 || value > base_n_max || n >= out_cap) {
                return;
            }
            for (int i = 0; i < n; ++i) {
                if (out[i] == value) {
                    return;
                }
            }
            out[n++] = value;
        };

        if (base_n_max >= 12) {
            add_unique(4);
            add_unique(8);
            add_unique(10);
            add_unique(12);
            add_unique(14);
            add_unique(base_n_max);
        } else {
            int probes[4];
            const int n_probes = profit_initial_probe_depths(base_n_max, probes, 4);
            for (int i = 0; i < n_probes; ++i) {
                add_unique(probes[i]);
            }
        }

        return n;
    }

    int profit_first_off_probe_depth(int base_n_max) const {
        int probes[8];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, 8);
        return n_probes > 0 ? probes[0] : 0;
    }

    int profit_next_off_probe_depth_after(int current_n, int base_n_max, bool * wrapped = nullptr) const {
        int probes[8];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, 8);
        if (n_probes <= 0) {
            if (wrapped) {
                *wrapped = false;
            }
            return 0;
        }
        for (int i = 0; i < n_probes; ++i) {
            if (probes[i] > current_n) {
                if (wrapped) {
                    *wrapped = false;
                }
                return probes[i];
            }
        }
        if (wrapped) {
            *wrapped = true;
        }
        return probes[0];
    }

    int profit_next_off_probe_depth(int base_n_max, int fallback_probe_n_max) const {
        if (base_n_max <= 0) {
            return 0;
        }

        if (profit_off_probe_next_n > 0) {
            return std::clamp<int>(profit_off_probe_next_n, 1, base_n_max);
        }

        const int first = profit_first_off_probe_depth(base_n_max);
        if (first > 0) {
            return first;
        }

        return std::clamp<int>(fallback_probe_n_max, 0, base_n_max);
    }

    void profit_prepare_off_probe_sweep(int base_n_max) {
        const int first = profit_first_off_probe_depth(base_n_max);
        profit_off_probe_next_n = first > 0 ? first : -1;
    }

    void profit_note_off_probe_result(int probed_n, int base_n_max, bool profitable) {
        if (profitable) {
            profit_off_probe_failures = 0;
            profit_prepare_off_probe_sweep(base_n_max);
            return;
        }

        bool wrapped = false;
        const int next = profit_next_off_probe_depth_after(probed_n, base_n_max, &wrapped);
        profit_off_probe_next_n = next > 0 ? next : -1;
        if (wrapped) {
            profit_off_probe_failures++;
        }
    }

    int profit_explore_depth_for_step(int current_n, int base_n_max, int explore_step) const {
        if (base_n_max <= 0) {
            return 0;
        }

        int probes[8];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, 8);
        if (n_probes <= 0) {
            return std::clamp<int>(current_n, 0, base_n_max);
        }

        int ordered[8];
        int n_ordered = 0;
        auto add = [&](int value) {
            if (value <= 0 || value > base_n_max || value == current_n || n_ordered >= 8) {
                return;
            }
            for (int i = 0; i < n_ordered; ++i) {
                if (ordered[i] == value) {
                    return;
                }
            }
            ordered[n_ordered++] = value;
        };

        for (int i = 0; i < n_probes; ++i) {
            if (probes[i] > current_n) {
                add(probes[i]);
            }
        }
        for (int i = 0; i < n_probes; ++i) {
            if (probes[i] < current_n) {
                add(probes[i]);
            }
        }

        if (n_ordered <= 0) {
            return std::clamp<int>(current_n, 0, base_n_max);
        }

        const int index = (std::max(1, explore_step) - 1) % n_ordered;
        return ordered[index];
    }

    int profit_next_unready_explore_depth(int current_n, int base_n_max, int explore_step) const {
        int probes[8];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, 8);
        for (int offset = 0; offset < n_probes; ++offset) {
            const int candidate = profit_explore_depth_for_step(
                    current_n,
                    base_n_max,
                    std::max(1, explore_step) + offset);
            if (candidate > 0 && !profit_ladder_probe_depth_ready(candidate)) {
                return candidate;
            }
        }
        return 0;
    }

    bool profit_ladder_probe_depth_ready(int depth) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= profit_initial_probe_min_samples() &&
            profit_depth[depth].cycle_ms > 0.0f &&
            profit_depth[depth].output_tokens > 0.0f;
    }

    int profit_candidate_min_samples() const {
        return std::max<int>(2, dm_profit_min_samples);
    }

    int profit_active_min_samples(int base_n_max) const {
        int probes[8];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, 8);
        return std::max<int>(dm_profit_min_samples, std::max(2, n_probes));
    }

    int profit_candidate_min_samples(int base_n_max) const {
        return std::max<int>(profit_candidate_min_samples(), profit_active_min_samples(base_n_max));
    }

    bool profit_candidate_ready_for_switch(int depth) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= profit_candidate_min_samples() &&
            profit_depth[depth].total_cycle_ms > 0.0 &&
            profit_depth[depth].total_output_tokens > 0.0;
    }

    bool profit_candidate_ready_for_switch(int depth, int base_n_max) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= profit_candidate_min_samples(base_n_max) &&
            profit_depth[depth].total_cycle_ms > 0.0 &&
            profit_depth[depth].total_output_tokens > 0.0;
    }

    void profit_begin_active_episode_from_stats(int depth, const profit_depth_stats & stats) {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            profit_active_episode_n = -1;
            profit_active_episode_start_samples = 0;
            profit_active_episode_start_output_tokens = 0.0;
            profit_active_episode_start_cycle_ms = 0.0;
            return;
        }

        profit_active_episode_n = depth;
        profit_active_episode_start_samples = stats.samples;
        profit_active_episode_start_output_tokens = stats.total_output_tokens;
        profit_active_episode_start_cycle_ms = stats.total_cycle_ms;
    }

    void profit_begin_active_episode(int depth) {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            profit_begin_active_episode_from_stats(-1, profit_depth_stats{});
            return;
        }
        profit_begin_active_episode_from_stats(depth, profit_depth[depth]);
    }

    bool profit_active_episode_ready(int depth, int base_n_max) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS || profit_active_episode_n != depth) {
            return false;
        }
        const profit_depth_stats & stats = profit_depth[depth];
        return stats.samples - profit_active_episode_start_samples >= profit_active_min_samples(base_n_max) &&
            stats.total_cycle_ms > profit_active_episode_start_cycle_ms &&
            stats.total_output_tokens > profit_active_episode_start_output_tokens;
    }

    float profit_active_episode_score(int depth) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS || profit_active_episode_n != depth) {
            return 0.0f;
        }
        const profit_depth_stats & stats = profit_depth[depth];
        const double output_tokens = stats.total_output_tokens - profit_active_episode_start_output_tokens;
        const double cycle_ms = stats.total_cycle_ms - profit_active_episode_start_cycle_ms;
        if (cycle_ms <= 0.0 || output_tokens <= 0.0 ||
                !std::isfinite(cycle_ms) || !std::isfinite(output_tokens)) {
            return 0.0f;
        }
        return server_adaptive_dm_score((float) output_tokens, (float) cycle_ms);
    }

    bool profit_initial_probe_set_ready(int base_n_max) const {
        return profit_baseline_ready() && profit_next_unready_probe_depth(base_n_max) < 0;
    }

    int profit_off_probe_interval() const {
        const int shift = std::clamp<int>(profit_off_probe_failures, 0, 6);
        const int multiplier = 1 << shift;
        const int base = std::max(1, (int) dm_probe_interval);
        if (base > INT_MAX / multiplier) {
            return INT_MAX;
        }
        return base * multiplier;
    }

    bool profit_should_probe_baseline() const {
        return dm_profit_baseline_interval > 0 &&
            profit_baseline_ready() &&
            !profit_baseline_probe_pending &&
            adaptive_n_max > 0 &&
            profit_cycles_since_baseline >= dm_profit_baseline_interval;
    }

    bool profit_expects_baseline_sample() const {
        return profit_baseline_probe_pending ||
            !profit_baseline_ready() ||
            adaptive_n_max == 0;
    }

    void profit_mark_baseline_probe(int resume_n = -1) {
        profit_baseline_probe_pending = true;
        profit_baseline_probe_resume_n = resume_n >= 0 ? resume_n : adaptive_n_max;
        adaptive_n_max = 0;
    }

    void apply_profit_recommendation(int recommended_n) {
        const int previous_n = adaptive_n_max;
        profit_last_recommended_n = recommended_n;
        adaptive_n_max = recommended_n;
        if (recommended_n > 0 && recommended_n != previous_n) {
            profit_begin_active_episode(recommended_n);
        } else if (recommended_n <= 0) {
            profit_begin_active_episode(-1);
        }
    }

    void observe_profit_acceptance(int n_draft, int n_accepted) {
        const int reached_positions = std::min<int>(n_draft, PROFIT_POSITIONS);
        for (int pos = 0; pos < reached_positions; ++pos) {
            const bool accepted = n_accepted > pos;
            const float sample = accepted ? 1.0f : 0.0f;
            server_adaptive_dm_ewma_init_or_update(profit_pos_accept_ewma[pos], sample, dm_profit_ewma_alpha,
                    profit_pos_samples[pos]);
            profit_pos_samples[pos]++;
        }
    }

    void observe_profit_timing(
            int requested_n_max,
            int actual_n_draft,
            int n_accepted,
            float draft_ms,
            float verify_ms,
            float accept_ms,
            float cycle_ms) {
        profit_depth_stats * dst = nullptr;
        if (requested_n_max <= 0) {
            dst = &profit_baseline;
        } else if (requested_n_max < PROFIT_DEPTHS) {
            dst = &profit_depth[requested_n_max];
        }
        if (!dst || cycle_ms <= 0.0f) {
            if (!dst && requested_n_max >= PROFIT_DEPTHS) {
                LOG_DBG("observe_profit_timing: requested_n_max=%d >= PROFIT_DEPTHS=%d, "
                        "timing data dropped\n", requested_n_max, PROFIT_DEPTHS);
            }
            return;
        }

        if (!std::isfinite(draft_ms) || !std::isfinite(verify_ms) ||
                !std::isfinite(accept_ms) || !std::isfinite(cycle_ms)) {
            return;
        }

        const float output_tokens = requested_n_max <= 0 ? 1.0f : 1.0f + (float) std::max(0, n_accepted);
        const float actual_draft_tokens = (float) std::max(0, actual_n_draft);

        if (requested_n_max > 0 &&
                requested_n_max == adaptive_n_max &&
                profit_last_recommended_n > 0 &&
                profit_active_episode_n != requested_n_max) {
            profit_begin_active_episode_from_stats(requested_n_max, *dst);
        }

        server_adaptive_dm_ewma_init_or_update(dst->draft_ms, draft_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->verify_ms, verify_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->accept_ms, accept_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->cycle_ms, cycle_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->output_tokens, output_tokens, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->actual_draft_tokens, actual_draft_tokens, dm_profit_ewma_alpha, dst->samples);
        dst->best_score = std::max(dst->best_score, server_adaptive_dm_score(output_tokens, cycle_ms));
        dst->total_output_tokens += (double) output_tokens;
        dst->total_cycle_ms += (double) cycle_ms;
        dst->samples++;

        if (requested_n_max <= 0) {
            profit_cycles_since_baseline = 0;
            profit_baseline_probe_pending = false;
        } else if (profit_baseline_ready()) {
            profit_cycles_since_baseline++;
        }
    }

    void observe_profit_timing(int n_draft, float draft_ms, float verify_ms, float accept_ms, float cycle_ms) {
        observe_profit_timing(n_draft, n_draft, 0, draft_ms, verify_ms, accept_ms, cycle_ms);
    }

    int decide_profit_n_max(int base_n_max) {
        if (base_n_max <= 0) {
            profit_last_recommended_n = 0;
            return 0;
        }

        const float baseline_score = profit_score_for_depth(0);
        if (!profit_baseline_ready()) {
            profit_current_score = 0.0f;
            profit_last_recommended_n = 0;
            return 0;
        }

        const int unready_probe = profit_next_unready_probe_depth(base_n_max);
        const bool collecting_initial_probe_set =
            profit_baseline_probe_resume_n <= 0 &&
            !profit_initial_probe_set_ready(base_n_max);
        if (collecting_initial_probe_set && unready_probe > 0) {
            profit_current_score = baseline_score;
            profit_last_recommended_n = unready_probe;
            return unready_probe;
        }

        const int current_n = profit_baseline_probe_resume_n > 0
            ? std::clamp<int>(profit_baseline_probe_resume_n, 0, base_n_max)
            : (adaptive_n_max < 0
                ? base_n_max
                : std::clamp<int>(adaptive_n_max, 0, base_n_max));
        const bool returning_from_baseline_probe = profit_baseline_probe_resume_n > 0;
        if (current_n == 0 && profit_baseline_probe_resume_n <= 0) {
            profit_current_score = baseline_score;
            profit_last_recommended_n = 0;
            return 0;
        }

        int candidates[PROFIT_CANDIDATES];
        int n_candidates = server_adaptive_dm_build_candidates(base_n_max, candidates, PROFIT_CANDIDATES);
        {
            bool found = false;
            for (int i = 0; i < n_candidates; i++) {
                if (candidates[i] == current_n) { found = true; break; }
            }
            if (!found && n_candidates < PROFIT_CANDIDATES && current_n > 0) {
                candidates[n_candidates++] = current_n;
                std::sort(candidates, candidates + n_candidates);
            }
        }

        bool ready[PROFIT_CANDIDATES] = {};
        bool estimated[PROFIT_CANDIDATES] = {};
        float expected_accept[PROFIT_CANDIDATES] = {};
        float cycle_ms[PROFIT_CANDIDATES] = {};
        float direct_score[PROFIT_CANDIDATES] = {};

        float current_score = 0.0f;
        bool current_ready = false;
        const bool baseline_ready = profit_baseline_ready();
        for (int i = 0; i < n_candidates; ++i) {
            const int n = candidates[i];
            if (n == 0) {
                ready[i] = baseline_ready;
                estimated[i] = false;
                expected_accept[i] = 0.0f;
                cycle_ms[i] = profit_baseline.cycle_ms;
                direct_score[i] = baseline_score;
            } else if (profit_depth_ready(n)) {
                bool positions_ready = false;
                const float survival_expected = server_adaptive_dm_survival_expected_accept(
                        profit_pos_accept_ewma, profit_pos_samples, PROFIT_POSITIONS,
                        n, dm_profit_min_samples, &positions_ready);
                const bool lower_from_current_survival = n < current_n && positions_ready;
                expected_accept[i] = lower_from_current_survival ?
                    survival_expected :
                    std::max(0.0f, profit_depth[n].output_tokens - 1.0f);
                ready[i] = true;
                estimated[i] = false;
                cycle_ms[i] = profit_depth[n].cycle_ms;
                direct_score[i] = lower_from_current_survival ?
                    server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]) :
                    profit_score_for_depth(n);
            } else if (n > 0) {
                // Find the ready depth closest to THIS candidate for extrapolation.
                int ref_depth = -1;
                int ref_dist  = INT_MAX;
                for (int d = 1; d < PROFIT_DEPTHS; ++d) {
                    if (profit_depth_ready(d)) {
                        const int dist = std::abs(d - n);
                        if (dist < ref_dist) { ref_dist = dist; ref_depth = d; }
                    }
                }
                if (ref_depth > 0) {
                    bool positions_ready = false;
                    expected_accept[i] = server_adaptive_dm_survival_expected_accept(
                            profit_pos_accept_ewma, profit_pos_samples, PROFIT_POSITIONS,
                            n, dm_profit_min_samples, &positions_ready);
                    if (positions_ready) {
                        const profit_depth_stats & ref = profit_depth[ref_depth];
                        const float draft_per_token = ref.draft_ms / (float) ref_depth;
                        const float est_cycle_ms = (float) n * draft_per_token + ref.verify_ms + ref.accept_ms;
                        if (est_cycle_ms > 0.0f) {
                            ready[i]     = true;
                            estimated[i] = true;
                            cycle_ms[i]  = est_cycle_ms;
                        }
                    }
                }
            }

            if (n == current_n && ready[i]) {
                current_ready = true;
                current_score = direct_score[i] > 0.0f ?
                    direct_score[i] :
                    server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
            }

            if (ready[i]) {
                const float score = direct_score[i] > 0.0f ?
                    direct_score[i] :
                    server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
                LOG_DBG("profit cand: n=%d ready=1 est=%d exp=%.2f cycle=%.1f score=%.2f%s\n",
                        n, estimated[i] ? 1 : 0,
                        expected_accept[i], cycle_ms[i], score,
                        n == current_n ? " (current)" : "");
            }
        }

        if (!current_ready && current_n > 0) {
            // estimate current_score so hysteresis is never bypassed
            int ref_depth = -1;
            int ref_dist  = INT_MAX;
            for (int d = 1; d < PROFIT_DEPTHS; ++d) {
                if (profit_depth_ready(d)) {
                    const int dist = std::abs(d - current_n);
                    if (dist < ref_dist) { ref_dist = dist; ref_depth = d; }
                }
            }
            if (ref_depth > 0) {
                bool pos_ready = false;
                const float ea = server_adaptive_dm_survival_expected_accept(
                        profit_pos_accept_ewma, profit_pos_samples, PROFIT_POSITIONS,
                        current_n, dm_profit_min_samples, &pos_ready);
                if (pos_ready) {
                    const profit_depth_stats & ref = profit_depth[ref_depth];
                    const float draft_per_token = ref.draft_ms / (float) ref_depth;
                    const float est_cycle = (float) current_n * draft_per_token + ref.verify_ms + ref.accept_ms;
                    if (est_cycle > 0.0f) {
                        current_ready = true;
                        current_score = server_adaptive_dm_score(1.0f + ea, est_cycle);
                    }
                }
            }
        }

        const bool evaluating_off_probe =
            profit_last_recommended_n == 0 &&
            current_n > 0 &&
            profit_baseline_probe_resume_n <= 0;
        if (evaluating_off_probe) {
            const float off_probe_score = profit_score_for_depth(current_n);
            const bool current_profitable =
                off_probe_score >= baseline_score * (1.0f + dm_profit_min);
            if (!current_profitable) {
                profit_note_off_probe_result(current_n, base_n_max, false);
                profit_consecutive_below_profit = 0;
                profit_current_score = off_probe_score > 0.0f ? off_probe_score : baseline_score;
                profit_last_recommended_n = 0;
                return 0;
            }
            profit_note_off_probe_result(current_n, base_n_max, true);
            profit_consecutive_below_profit = 0;
            profit_current_score = off_probe_score;
            profit_last_recommended_n = current_n;
            return current_n;
        }

        auto best = server_adaptive_dm_best_profit_candidate(
                candidates, n_candidates, ready, estimated, expected_accept, cycle_ms);
        server_adaptive_dm_profit_candidate best_direct;
        for (int i = 0; i < n_candidates; ++i) {
            if (!ready[i] || estimated[i]) {
                continue;
            }
            const float score = direct_score[i] > 0.0f ?
                direct_score[i] :
                server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
            if (!best_direct.ready || score > best_direct.score) {
                best_direct.ready = true;
                best_direct.n = candidates[i];
                best_direct.score = score;
                best_direct.estimated = false;
            }
        }
        if (best_direct.ready) {
            best = best_direct;
        }
        if (!best.ready) {
            profit_current_score = current_score;
            profit_last_recommended_n = current_n;
            return current_n;
        }

        int recommended = best.n;
        const bool estimated_demote =
            best.estimated &&
            best.n > 0 &&
            best.n < current_n &&
            current_ready;
        const bool active_episode_ready =
            current_n > 0 &&
            profit_active_episode_ready(current_n, base_n_max);
        const float active_episode_score =
            active_episode_ready ? profit_active_episode_score(current_n) : 0.0f;
        const bool baseline_wins =
            best.n == 0 ||
            (best.n > 0 && best.score < baseline_score * (1.0f + dm_profit_min));
        if (estimated_demote && (!baseline_ready ||
                    current_score >= baseline_score * (1.0f + dm_profit_min))) {
            recommended = current_n;
        } else if (baseline_ready && baseline_wins) {
            if (current_n > 0 &&
                    !returning_from_baseline_probe &&
                    !active_episode_ready) {
                recommended = current_n;
                profit_consecutive_below_profit = 0;
                profit_current_score = current_ready ? current_score : best.score;
                profit_last_recommended_n = recommended;
                profit_baseline_probe_resume_n = -1;
                return recommended;
            }
            if (current_n > 0 &&
                    active_episode_ready &&
                    active_episode_score >= baseline_score * (1.0f + dm_profit_min)) {
                recommended = current_n;
                profit_consecutive_below_profit = 0;
                profit_current_score = active_episode_score;
                profit_last_recommended_n = recommended;
                profit_baseline_probe_resume_n = -1;
                return recommended;
            }
            profit_consecutive_below_profit++;
            const bool disable_now =
                returning_from_baseline_probe ||
                profit_consecutive_below_profit >= dm_off_dwell;
            recommended = disable_now ? 0 : current_n;
            if (disable_now && current_n > 0) {
                profit_prepare_off_probe_sweep(base_n_max);
                LOG_INF("adaptive-dm: disabling speculative depth "
                        "(best_score=%.3f baseline=%.3f profit_min=%.3f below_count=%d)\n",
                        best.score, baseline_score,
                        dm_profit_min, profit_consecutive_below_profit);
            }
        } else {
            profit_consecutive_below_profit = 0;
            profit_off_probe_failures = 0;
            const bool positive_switch =
                current_n > 0 &&
                best.n > 0 &&
                best.n != current_n;
            const bool unconfirmed_positive_switch =
                positive_switch &&
                !best.estimated &&
                !profit_candidate_ready_for_switch(best.n, base_n_max);
            if (positive_switch && !active_episode_ready) {
                recommended = current_n;
            } else if (unconfirmed_positive_switch) {
                recommended = current_n;
            } else {
                recommended = server_adaptive_dm_apply_profit_hysteresis(
                        current_n,
                        best.n,
                        current_ready ? current_score : 0.0f,
                        best.score,
                        dm_profit_raise_margin,
                        dm_profit_lower_margin,
                        candidates,
                        n_candidates);
            }
        }

        const bool positive_demotion =
            current_n > 0 &&
            recommended > 0 &&
            recommended < current_n;
        if (positive_demotion &&
                baseline_ready &&
                !returning_from_baseline_probe &&
                !profit_baseline_probe_pending &&
                dm_profit_baseline_interval > 0 &&
                profit_cycles_since_baseline >= dm_profit_baseline_interval) {
            profit_mark_baseline_probe(current_n);
            profit_current_score = current_ready ? current_score : best.score;
            profit_last_recommended_n = 0;
            return 0;
        }

        recommended = std::clamp(recommended, 0, base_n_max);
        profit_current_score = best.score;
        profit_last_recommended_n = recommended;
        profit_baseline_probe_resume_n = -1;

        LOG_DBG("profit decide: current_n=%d best_n=%d rec=%d "
                "best_ready=%d best_est=%d best_score=%.3f "
                "curr_ready=%d curr_score=%.3f base_n=%d\n",
                current_n, best.n, recommended,
                best.ready ? 1 : 0, best.estimated ? 1 : 0,
                best.score,
                current_ready ? 1 : 0, current_score, base_n_max);

        return recommended;
    }

    fringe_decision fringe_decision_at_window(int n_max, int window, int min_reach) const {
        fringe_decision result;
        if (n_max <= 0 || window <= 0) {
            return result;
        }

        const int window_start = std::max(0, n_max - window);
        for (int pos = window_start; pos < n_max; pos++) {
            for (int i = 0; i < fringe_ring_count; i++) {
                const int idx = (fringe_ring_idx - 1 - i + FRINGE_WINDOW) % FRINGE_WINDOW;
                const auto & e = fringe_ring[idx];
                if (e.epoch != fringe_epoch) {
                    continue;
                }
                if (e.n_draft > pos) {
                    result.reached++;
                    if (e.n_accepted > pos) {
                        result.accepted++;
                    }
                }
            }
        }

        result.fringe = result.reached > 0 ? (float) result.accepted / result.reached : 0.0f;

        if (result.reached < min_reach) {
            int epoch_reached = 0;
            int epoch_accepted = 0;
            for (int pos = window_start; pos < n_max; pos++) {
                if (pos < FRINGE_REACH_POSITIONS) {
                    epoch_reached += fringe_epoch_reached[pos];
                    epoch_accepted += fringe_epoch_accepted[pos];
                }
            }
            if (epoch_reached >= min_reach) {
                result.reached = epoch_reached;
                result.accepted = epoch_accepted;
                result.fringe = (float) epoch_accepted / epoch_reached;
            }
        }

        return result;
    }
};
