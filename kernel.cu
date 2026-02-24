__global__ void fused_bn_leakyrelu_kernel(
    const float* input,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int C,
    float eps,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * C) {
        int c = idx % C;
        
        float normalized = (input[idx] - mean[c]) / sqrtf(var[c] + eps);
        float scaled = gamma[c] * normalized + beta[c];
        
        output[idx] = scaled > 0.0f ? scaled : alpha * scaled;
    }
}
