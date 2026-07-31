#include <stdio.h>
#include <math.h>
#include <string.h>

// Disable double-promotion warnings for this test (uses float extensively)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"

#include "ggml.h"

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void quantize_row_tq3_1s_ref(const float * x, void * y, long long k);
extern void dequantize_row_tq3_1s(const void * x, float * y, long long k);
extern void quantize_row_tq4_1s_ref(const float * x, void * y, long long k);
extern void dequantize_row_tq4_1s(const void * x, float * y, long long k);

static double cosine_similarity(const float * x, const float * y, int n) {
    double dot = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    for (int i = 0; i < n; i++) {
        dot += x[i] * y[i];
        nx += x[i] * x[i];
        ny += y[i] * y[i];
    }
    return nx > 0.0 && ny > 0.0 ? dot / sqrt(nx) / sqrt(ny) : 0.0;
}

int main(void) {
    const int d = 128;
    char buf[256];
    float input[128], output[128];
    double mse, cosv, ni, no;
    int failed = 0;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    if (GGML_TYPE_TURBO3_TCQ != 45 || GGML_TYPE_TURBO2_TCQ != 46 ||
        GGML_TYPE_TQ3_1S != 47 || GGML_TYPE_TQ4_1S != 48) {
        printf("enum separation FAILED: turbo3_tcq=%d turbo2_tcq=%d tq3_1s=%d tq4_1s=%d\n",
                GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TQ3_1S, GGML_TYPE_TQ4_1S);
        failed++;
    }

    if (ggml_type_size(GGML_TYPE_TQ3_1S) != 16 || ggml_type_size(GGML_TYPE_TQ4_1S) != 20) {
        printf("TQ type sizes FAILED: tq3_1s=%zu tq4_1s=%zu\n",
                ggml_type_size(GGML_TYPE_TQ3_1S), ggml_type_size(GGML_TYPE_TQ4_1S));
        failed++;
    }

    /* Test 1: basis vector */
    memset(input, 0, sizeof(input));
    input[0] = 1.0;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    printf("  In:  [%.6f, %.6f, %.6f, %.6f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.6f, %.6f, %.6f, %.6f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f OutNorm=%.6f\n\n", mse/d, ni > 0 && no > 0 ? cosv/(double)sqrt(ni)/(double)sqrt(no) : 0, (double)sqrt(no));

    /* Test 2: large-norm vector */
    for (int i = 0; i < d; i++) input[i] = sin(i*0.1+0.5) * 10.0;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    printf("Test 2 (turbo3): sin*10\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f InNorm=%.2f OutNorm=%.2f\n\n", mse/d, cosv/(double)sqrt(ni)/(double)sqrt(no), (double)sqrt(ni), (double)sqrt(no));

    /* Test 3: turbo4 */
    for (int i = 0; i < d; i++) input[i] = cos(i*0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    printf("Test 3 (turbo4): cos*5\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f\n\n", mse/d, cosv/(double)sqrt(ni)/(double)sqrt(no));

    /* Test 4: tq3_1s weight format */
    for (int i = 0; i < 32; i++) input[i] = sin(i*0.17f) * 3.0f + cos(i*0.07f);
    quantize_row_tq3_1s_ref(input, buf, 32);
    dequantize_row_tq3_1s(buf, output, 32);
    cosv = cosine_similarity(input, output, 32);
    printf("Test 4 (tq3_1s): cosine=%.6f\n", cosv);
    if (cosv < 0.96f) {
        printf("  FAILED: tq3_1s cosine below threshold\n");
        failed++;
    }

    /* Test 5: tq4_1s weight format */
    for (int i = 0; i < 32; i++) input[i] = cos(i*0.19f) * 2.5f - sin(i*0.11f);
    quantize_row_tq4_1s_ref(input, buf, 32);
    dequantize_row_tq4_1s(buf, output, 32);
    cosv = cosine_similarity(input, output, 32);
    printf("Test 5 (tq4_1s): cosine=%.6f\n\n", cosv);
    if (cosv < 0.985f) {
        printf("  FAILED: tq4_1s cosine below threshold\n");
        failed++;
    }

    printf("=== Done ===\n");
    return failed > 0;
}

#pragma GCC diagnostic pop
