#include "common.h"
#include "speculative.h"

#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static void test(void) {
    printf("test-speculative: testing null safety\n\n");

    // Test 1: null reset should not crash
    printf("test-speculative: test null reset\n");
    common_speculative_reset(nullptr, 0);
    printf("  OK: null reset did not crash\n");

    // Test 2: null begin should not crash
    printf("test-speculative: test null begin\n");
    llama_tokens prompt = {1, 2, 3};
    common_speculative_begin(nullptr, 0, prompt);
    printf("  OK: null begin did not crash\n");

    printf("test-speculative: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-speculative: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
