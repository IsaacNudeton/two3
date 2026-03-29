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
