/*
 * two3.h — {2,3} Computing Kernel
 * Isaac Oravec & Claude | XYZT Computing Project | 2026
 *
 * The structural atom: two entities transacting through a substrate.
 *   +1  = entity A (add)
 *   -1  = entity B (subtract)
 *    0  = substrate (the wire that doesn't fire)
 *
 * No floating-point multiply in the forward pass.
 * Weights are 2 bits. Matmul is addition and subtraction.
 * The zeros aren't dead — they're the topology.
 *
 * On custom hardware (FPGA/ASIC), the "multiply by sign" becomes
 * a 2-input mux: pass, negate, or zero. No ALU needed.
 * On CUDA, we use integer multiply by {-1,0,+1} which is 1 cycle.
 *
 * Weight encoding (2 bits per weight, 4 packed per byte, LSB-first):
 *   0b00 = 0   substrate
 *   0b01 = +1  entity A
 *   0b10 = -1  entity B
 *   0b11 = reserved (never generated)
 */

#ifndef TWO3_H
#define TWO3_H

#include <stdint.h>
#include <stddef.h>

/* ----------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------- */

#define TWO3_ENCODE_ZERO   0x00   /* substrate  */
#define TWO3_ENCODE_PLUS   0x01   /* entity A   */
#define TWO3_ENCODE_MINUS  0x02   /* entity B   */
#define TWO3_WEIGHTS_PER_BYTE  4  /* 2 bits each */

/* CUDA block dimensions — tuned for Turing SM 7.5 (RTX 2080 Super) */
#define TWO3_BLOCK_X  16
#define TWO3_BLOCK_Y  16

/* ----------------------------------------------------------------
 * Structures
 * ---------------------------------------------------------------- */

/* Packed ternary weight matrix */
typedef struct {
    uint8_t* packed;       /* [rows, cols/4] device memory, 4 weights per byte */
    float    scale;        /* absmean of original float weights               */
    int      rows;         /* output features (M)                             */
    int      cols;         /* input features  (K), must be divisible by 4     */
} Two3Weights;

/* Quantized activation matrix */
typedef struct {
    int8_t*  data;         /* [tokens, dim] device memory, int8 per element */
    float*   scales;       /* [tokens] device memory, one absmax scale per token */
    int      tokens;       /* sequence length or batch*seq (S) */
    int      dim;          /* feature dimension (K)            */
} Two3Activations;

/* Output accumulator */
typedef struct {
    int32_t* acc;          /* [tokens, out_dim] device memory, int32 accumulators */
    int      tokens;
    int      out_dim;
} Two3Output;

/* ----------------------------------------------------------------
 * Host API — called from C, launches CUDA internally
 * ---------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Weight management ----- */

/* Quantize float weights to ternary and pack.
 * w_float: [rows, cols] host memory, row-major
 * Returns packed weights in device memory. */
Two3Weights two3_pack_weights(const float* w_float, int rows, int cols);

/* Free device memory for packed weights. */
void two3_free_weights(Two3Weights* w);

/* ----- Activation quantization ----- */

/* Quantize float activations to int8 using per-token absmax.
 * x_float: [tokens, dim] host memory, row-major
 * Returns quantized activations in device memory. */
Two3Activations two3_quantize_acts(const float* x_float, int tokens, int dim);

/* Free device memory for quantized activations. */
void two3_free_acts(Two3Activations* a);

/* ----- Forward pass: the {2,3} kernel ----- */

/* Y = X @ W^T
 * Computes: Y[s][m] = sum_k  sign(W[m][k]) * X[s][k]
 * All integer. No float multiply.
 * Returns int32 accumulators in device memory. */
Two3Output two3_forward(const Two3Weights* W, const Two3Activations* X);

/* Dequantize int32 output to float.
 * Applies: y_float = acc * act_scale * weight_scale
 * y_float: [tokens, out_dim] host memory (caller allocates) */
void two3_dequantize_output(const Two3Output* Y,
                            const Two3Weights* W,
                            const Two3Activations* X,
                            float* y_float);

/* Free device memory for output. */
void two3_free_output(Two3Output* y);

/* ----- Backward pass ----- */

/* Pre-allocated device buffers for backward pass.
 * Init once, reuse across all backward calls per training step.
 * Eliminates cudaMalloc/cudaFree overhead. */
typedef struct {
    /* Ternary projection backward buffers */
    float *d_dY;        /* [max_M] device */
    float *d_X;         /* [max_K] device */
    float *d_dX;        /* [max_K] device */
    float *d_W_latent;  /* [max_M * max_K] device */
    float *d_dW;        /* [max_M * max_K] device */
    float *h_dX_tmp;    /* [max_K] host scratch for accumulate readback */
    int    max_M;
    int    max_K;

    /* Attention backward buffers (sized at init for max_seq × D/KV) */
    float *d_attn_Q;       /* [max_seq * D] device */
    float *d_attn_K;       /* [max_seq * KV] device */
    float *d_attn_V;       /* [max_seq * KV] device */
    float *d_attn_dout;    /* [max_seq * D] device */
    float *d_attn_dQ;      /* [max_seq * D] device */
    float *d_attn_dK;      /* [max_seq * KV] device */
    float *d_attn_dV;      /* [max_seq * KV] device */
    float *d_attn_scratch; /* [n_heads * max_seq * max_seq] device */
    float *h_attn_dK_tmp;  /* [max_seq * KV] host scratch */
    float *h_attn_dV_tmp;  /* [max_seq * KV] host scratch */
    int    attn_max_seq;
    int    attn_D;
    int    attn_KV;
    int    attn_n_heads;
} Two3BackwardCtx;

/* Allocate backward context sized for max(M) and max(K) across all layers.
 * Also allocates attention backward buffers if max_seq > 0.
 * Call once at training init. */
Two3BackwardCtx two3_backward_ctx_init(int max_M, int max_K,
                                        int max_seq, int D, int KV, int n_heads);

/* Free backward context. */
void two3_backward_ctx_free(Two3BackwardCtx* ctx);

/* Backward with pre-allocated context (fast path — no per-call malloc).
 * S = number of tokens (1 for single-vector, seq_len for batched). */
void two3_backward_fast(
    Two3BackwardCtx* ctx,
    const Two3Weights* W,     /* packed ternary (device) */
    const float* dY_host,     /* [S, M] */
    const float* X_host,      /* [S, K] */
    const float* W_latent,    /* [M, K] host */
    float* dX_host,           /* [S, K] accumulate */
    float* dW_host,           /* [M, K] accumulate */
    int S, int M, int K,
    float ste_clip);

/* ----- Attention backward on GPU ----- */

/* Batched causal attention backward with pre-allocated buffers (fast path). */
void two3_attention_backward_fast(
    Two3BackwardCtx* ctx,
    const float* Q_host,     /* [seq_len, n_heads * head_dim] */
    const float* K_host,     /* [seq_len, n_kv_heads * head_dim] */
    const float* V_host,     /* [seq_len, n_kv_heads * head_dim] */
    const float* d_out_host, /* [seq_len, n_heads * head_dim] */
    float* dQ_host,          /* [seq_len, n_heads * head_dim] OUTPUT */
    float* dK_host,          /* [seq_len, n_kv_heads * head_dim] ACCUMULATE */
    float* dV_host,          /* [seq_len, n_kv_heads * head_dim] ACCUMULATE */
    int seq_len, int n_heads, int n_kv_heads, int head_dim);

/* Original attention backward (allocates per call — kept for compat). */
void two3_attention_backward(
    const float* Q_host,     /* [seq_len, n_heads * head_dim] */
    const float* K_host,     /* [seq_len, n_kv_heads * head_dim] */
    const float* V_host,     /* [seq_len, n_kv_heads * head_dim] */
    const float* d_out_host, /* [seq_len, n_heads * head_dim] */
    float* dQ_host,          /* [seq_len, n_heads * head_dim] OUTPUT */
    float* dK_host,          /* [seq_len, n_kv_heads * head_dim] ACCUMULATE */
    float* dV_host,          /* [seq_len, n_kv_heads * head_dim] ACCUMULATE */
    int seq_len, int n_heads, int n_kv_heads, int head_dim);

/* Original backward (allocates per call — kept for tests/compat). */
void two3_backward(
    const Two3Weights* W,     /* packed ternary (device) */
    const float* dY_host,     /* [1, M] */
    const float* X_host,      /* [1, K] */
    const float* W_latent,    /* [M, K] host */
    float* dX_host,           /* [K] accumulate */
    float* dW_host,           /* [M, K] accumulate */
    int M, int K,
    float ste_clip);

/* ----- Utilities ----- */

/* Reference float matmul on CPU for verification.
 * C[s][m] = sum_k X[s][k] * W[m][k]
 * All host memory, row-major. */
void two3_ref_matmul(const float* X, const float* W, float* C,
                     int S, int K, int M);

/* Print weight statistics: % substrate, % entity A, % entity B */
void two3_print_stats(const Two3Weights* w);

#ifdef __cplusplus
}
#endif

#endif /* TWO3_H */
