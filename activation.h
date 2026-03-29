/*
 * activation.h — Squared ReLU for {2,3} Architecture
 *
 * y = (x > 0) ? x² : 0
 *
 * Why squared: standard ReLU is linear for positive values.
 * Squared ReLU adds nonlinearity without a lookup table.
 * On ternary: the input comes from a ternary matmul (int32 acc),
 * gets dequantized to float, then squared ReLU, then re-quantized
 * for the next ternary matmul. The squaring is the ONLY nonlinear
 * operation in the forward pass.
 *
 * Isaac & CC — March 2026
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#ifdef __CUDACC__

__global__ void kernel_squared_relu(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = (v > 0.0f) ? v * v : 0.0f;
}

/* Backward: dy/dx = 2x for x > 0, else 0 */
__global__ void kernel_squared_relu_backward(
    float *dx, const float *dy, const float *x, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    dx[i] = (v > 0.0f) ? dy[i] * 2.0f * v : 0.0f;
}

#endif /* __CUDACC__ */

/* CPU reference */
static void squared_relu_cpu(float *y, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = (v > 0.0f) ? v * v : 0.0f;
    }
}

#endif /* ACTIVATION_H */
