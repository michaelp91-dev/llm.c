/*
RoPE: Rotary Position Embeddings
Applied to Q and K after the QKV projection / permute, before attention scores.
No learned position embedding table (wpe) is used.

Layout expected for q/k: (B, NH, T, HD) where HD is head dimension (must be even).
cos/sin tables: (maxT, HD/2) stored as float for numerical stability.
*/
#pragma once
#include <math.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// Precompute cos/sin tables on CPU then copy to GPU (called once at startup)

inline void rope_precompute_cpu(float* cos_table, float* sin_table, int max_seq_len, int head_dim, float theta = 10000.0f) {
    assert(head_dim % 2 == 0);
    int half = head_dim / 2;
    for (int t = 0; t < max_seq_len; t++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / (float)head_dim);
            float angle = (float)t * freq;
            cos_table[t * half + i] = cosf(angle);
            sin_table[t * half + i] = sinf(angle);
        }
    }
}

// ----------------------------------------------------------------------------
// Forward: rotate Q and K in-place
// q, k: (B, NH, T, HD)
// cos, sin: (maxT, HD/2)  -- only first T rows used

__global__ void rope_forward_kernel(floatX* q, floatX* k,
                                    const float* cos, const float* sin,
                                    int B, int NH, int T, int HD) {
    // one thread per (b, nh, t, pair)
    int pair = HD / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * pair;
    if (idx >= total) return;

    int b = idx / (NH * T * pair);
    int rest = idx % (NH * T * pair);
    int nh = rest / (T * pair);
    rest = rest % (T * pair);
    int t = rest / pair;
    int p = rest % pair;  // which pair in the head dim

    int base = ((b * NH + nh) * T + t) * HD;
    int i0 = base + 2 * p;
    int i1 = base + 2 * p + 1;

    float c = cos[t * pair + p];
    float s = sin[t * pair + p];

    float q0 = (float)q[i0];
    float q1 = (float)q[i1];
    q[i0] = (floatX)(q0 * c - q1 * s);
    q[i1] = (floatX)(q0 * s + q1 * c);

    float k0 = (float)k[i0];
    float k1 = (float)k[i1];
    k[i0] = (floatX)(k0 * c - k1 * s);
    k[i1] = (floatX)(k0 * s + k1 * c);
}

void rope_forward(floatX* q, floatX* k, const float* cos, const float* sin,
                  int B, int NH, int T, int HD, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(HD % 2 == 0);
    int pair = HD / 2;
    int total = B * NH * T * pair;
    int block_size = 256;
    int grid = CEIL_DIV(total, block_size);
    rope_forward_kernel<<<grid, block_size, 0, stream>>>(q, k, cos, sin, B, NH, T, HD);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Backward: conjugate rotation on dQ and dK (same as forward with -sin)
// dq, dk: (B, NH, T, HD) gradients w.r.t. rotated Q/K; we rotate them back in-place

__global__ void rope_backward_kernel(floatX* dq, floatX* dk,
                                     const float* cos, const float* sin,
                                     int B, int NH, int T, int HD) {
    int pair = HD / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * pair;
    if (idx >= total) return;

    int b = idx / (NH * T * pair);
    int rest = idx % (NH * T * pair);
    int nh = rest / (T * pair);
    rest = rest % (T * pair);
    int t = rest / pair;
    int p = rest % pair;

    int base = ((b * NH + nh) * T + t) * HD;
    int i0 = base + 2 * p;
    int i1 = base + 2 * p + 1;

    float c = cos[t * pair + p];
    float s = sin[t * pair + p];

    // conjugate: use -sin
    float dq0 = (float)dq[i0];
    float dq1 = (float)dq[i1];
    dq[i0] = (floatX)( dq0 * c + dq1 * s);
    dq[i1] = (floatX)(-dq0 * s + dq1 * c);

    float dk0 = (float)dk[i0];
    float dk1 = (float)dk[i1];
    dk[i0] = (floatX)( dk0 * c + dk1 * s);
    dk[i1] = (floatX)(-dk0 * s + dk1 * c);
}

void rope_backward(floatX* dq, floatX* dk, const float* cos, const float* sin,
                   int B, int NH, int T, int HD, cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(HD % 2 == 0);
    int pair = HD / 2;
    int total = B * NH * T * pair;
    int block_size = 256;
    int grid = CEIL_DIV(total, block_size);
    rope_backward_kernel<<<grid, block_size, 0, stream>>>(dq, dk, cos, sin, B, NH, T, HD);
    cudaCheck(cudaGetLastError());
}
