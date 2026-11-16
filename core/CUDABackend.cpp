#include "CUDABackend.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_cublasHandle = nullptr;
static bool g_cublasInitialized = false;

static cublasHandle_t getCublasHandle() {
    if (!g_cublasInitialized) {
        cublasStatus_t status = cublasCreate(&g_cublasHandle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize cuBLAS handle");
        }
        g_cublasInitialized = true;
    }
    return g_cublasHandle;
}

// -------------------------------------------------------------
// CUDA Kernels
// -------------------------------------------------------------

__global__ void k_fill(float* data, float val, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

__global__ void k_accumulate(float* dst, const float* src, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

__global__ void k_add_bias_2d(float* data, const float* bias, int batchSize, int features) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batchSize * features;
    if (idx < total) {
        int feat = idx % features;
        data[idx] += bias[feat];
    }
}

__global__ void k_sum_rows(const float* gradOut, float* gradBias, int rows, int cols) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; ++r) {
            sum += gradOut[r * cols + c];
        }
        gradBias[c] += sum;
    }
}

__global__ void k_relu_forward(const float* in, float* out, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
    }
}

__global__ void k_relu_backward(const float* in, const float* gradOut, float* gradIn, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        gradIn[idx] = in[idx] > 0.0f ? gradOut[idx] : 0.0f;
    }
}

__global__ void k_maxpool2d_forward(const float* in, float* out, int* mask,
                                    int N, int C, int H, int W,
                                    int outH, int outW,
                                    int poolH, int poolW, int strideH, int strideW) {
    int total = N * C * outH * outW;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % outW;
    int rem = idx / outW;
    int oh = rem % outH;
    rem = rem / outH;
    int c = rem % C;
    int n = rem / C;

    int hStart = oh * strideH;
    int wStart = ow * strideW;

    const float* inChannel = in + (n * C + c) * H * W;
    float maxVal = -1e30f;
    int maxIdx = -1;

    for (int ph = 0; ph < poolH; ++ph) {
        int ih = hStart + ph;
        if (ih >= H) continue;
        for (int pw = 0; pw < poolW; ++pw) {
            int iw = wStart + pw;
            if (iw >= W) continue;
            int curInIdx = ih * W + iw;
            float val = inChannel[curInIdx];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = curInIdx;
            }
        }
    }

    out[idx] = maxVal;
    mask[idx] = maxIdx;
}

__global__ void k_maxpool2d_backward(const float* gradOut, const int* mask, float* gradIn,
                                     int N, int C, int H, int W,
                                     int outH, int outW) {
    int total = N * C * outH * outW;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int rem = idx / (outH * outW);
    int c = rem % C;
    int n = rem / C;

    int maxIdx = mask[idx];
    if (maxIdx >= 0 && maxIdx < H * W) {
        atomicAdd(&gradIn[(n * C + c) * H * W + maxIdx], gradOut[idx]);
    }
}

__global__ void k_global_avg_pool_forward(const float* in, float* out,
                                         int N, int C, int H, int W) {
    int total = N * C;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    const float* channelPtr = in + idx * H * W;
    float sum = 0.0f;
    int hw = H * W;
    for (int i = 0; i < hw; ++i) {
        sum += channelPtr[i];
    }
    out[idx] = sum / static_cast<float>(hw);
}

__global__ void k_global_avg_pool_backward(const float* gradOut, float* gradIn,
                                          int N, int C, int H, int W) {
    int total = N * C * H * W;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int hw = H * W;
    int nc = idx / hw;
    gradIn[idx] = gradOut[nc] / static_cast<float>(hw);
}

__global__ void k_im2col(const float* data_im, float* data_col,
                         int channels, int height, int width,
                         int kernel_h, int kernel_w,
                         int pad_h, int pad_w,
                         int stride_h, int stride_w,
                         int height_col, int width_col) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int total = channels * height_col * width_col;
    if (index >= total) return;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int channel_in = index / (width_col * height_col);

    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    int col_offset = (channel_out * height_col + h_out) * width_col + w_out;
    int im_channel_offset = channel_in * height * width;
    int spatial_col_size = height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
        int r = h_in + i;
        for (int j = 0; j < kernel_w; ++j) {
            int c = w_in + j;
            float val = 0.0f;
            if (r >= 0 && r < height && c >= 0 && c < width) {
                val = data_im[im_channel_offset + r * width + c];
            }
            data_col[col_offset] = val;
            col_offset += spatial_col_size;
        }
    }
}

__global__ void k_col2im(const float* data_col, float* data_im,
                         int channels, int height, int width,
                         int kernel_h, int kernel_w,
                         int pad_h, int pad_w,
                         int stride_h, int stride_w,
                         int height_col, int width_col) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int total = channels * height * width;
    if (index >= total) return;

    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);

    float val = 0.0f;
    int spatial_col_size = height_col * width_col;

    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_padded = h + pad_h - kh;
        if (h_padded < 0 || h_padded % stride_h != 0) continue;
        int oh = h_padded / stride_h;
        if (oh >= height_col) continue;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_padded = w + pad_w - kw;
            if (w_padded < 0 || w_padded % stride_w != 0) continue;
            int ow = w_padded / stride_w;
            if (ow >= width_col) continue;

            int k_row = (c * kernel_h + kh) * kernel_w + kw;
            int s_col = oh * width_col + ow;
            val += data_col[k_row * spatial_col_size + s_col];
        }
    }
    data_im[index] = val;
}

__global__ void k_add_bias_channel(float* output, const float* bias, int total, int outC, int spatial) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / spatial) % outC;
    output[idx] += bias[c];
}

__global__ void k_conv2d_bias_backward(const float* gradOut, float* gradBias, int N, int outC, int spatial) {
    int c = blockIdx.x;
    if (c >= outC) return;

    int total_per_channel = N * spatial;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < total_per_channel; i += blockDim.x) {
        int n = i / spatial;
        int s = i % spatial;
        local_sum += gradOut[(n * outC + c) * spatial + s];
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        gradBias[c] = sdata[0];
    }
}

static float* g_colWorkspace = nullptr;
static size_t g_colWorkspaceSize = 0;

static float* getColWorkspace(size_t requiredElements) {
    if (requiredElements > g_colWorkspaceSize) {
        if (g_colWorkspace) {
            cudaFree(g_colWorkspace);
            g_colWorkspace = nullptr;
            g_colWorkspaceSize = 0;
        }
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&g_colWorkspace), requiredElements * sizeof(float)));
        g_colWorkspaceSize = requiredElements;
    }
    return g_colWorkspace;
}

__global__ void k_batchnorm2d_forward_train(const float* in, float* out,
                                            const float* gamma, const float* beta,
                                            float* runningMean, float* runningVar,
                                            float* saveMean, float* saveInvVar,
                                            int N, int C, int H, int W,
                                            float eps, float momentum) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int hw = H * W;
    int M = N * hw;

    __shared__ float s_sum[256];
    __shared__ float s_sumSq[256];
    __shared__ float s_mean;
    __shared__ float s_invVar;

    float threadSum = 0.0f;
    for (int idx = tid; idx < M; idx += blockDim.x) {
        int n = idx / hw;
        int i = idx % hw;
        threadSum += in[(n * C + c) * hw + i];
    }
    s_sum[tid] = threadSum;
    __syncthreads();

    // Reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_mean = s_sum[0] / static_cast<float>(M);
    }
    __syncthreads();

    float mean = s_mean;
    float threadSumSq = 0.0f;
    for (int idx = tid; idx < M; idx += blockDim.x) {
        int n = idx / hw;
        int i = idx % hw;
        float diff = in[(n * C + c) * hw + i] - mean;
        threadSumSq += diff * diff;
    }
    s_sumSq[tid] = threadSumSq;
    __syncthreads();

    // Reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sumSq[tid] += s_sumSq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float var = s_sumSq[0] / static_cast<float>(M);
        float invVar = rsqrtf(var + eps);
        s_invVar = invVar;
        if (saveMean) saveMean[c] = mean;
        if (saveInvVar) saveInvVar[c] = invVar;
        if (runningMean) runningMean[c] = (1.0f - momentum) * runningMean[c] + momentum * mean;
        if (runningVar) runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * var;
    }
    __syncthreads();

    float invVar = s_invVar;
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    for (int idx = tid; idx < M; idx += blockDim.x) {
        int n = idx / hw;
        int i = idx % hw;
        int inIdx = (n * C + c) * hw + i;
        out[inIdx] = g * (in[inIdx] - mean) * invVar + b;
    }
}

__global__ void k_batchnorm2d_forward_eval(const float* in, float* out,
                                           const float* gamma, const float* beta,
                                           const float* runningMean, const float* runningVar,
                                           int total, int C, int H, int W, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total) {
        int hw = H * W;
        int rem = idx / hw;
        int c = rem % C;
        float mean = runningMean ? runningMean[c] : 0.0f;
        float var = runningVar ? runningVar[c] : 1.0f;
        float invVar = rsqrtf(var + eps);
        float g = gamma ? gamma[c] : 1.0f;
        float b = beta ? beta[c] : 0.0f;
        out[idx] = g * (in[idx] - mean) * invVar + b;
    }
}

__global__ void k_batchnorm2d_backward(const float* gradOut, const float* in, float* gradIn,
                                       const float* gamma, float* gradGamma, float* gradBeta,
                                       const float* saveMean, const float* saveInvVar,
                                       int N, int C, int H, int W) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int hw = H * W;
    int M = N * hw;

    __shared__ float s_sumGrad[256];
    __shared__ float s_sumGradXhat[256];

    float mean = saveMean[c];
    float invVar = saveInvVar[c];
    float g = gamma ? gamma[c] : 1.0f;

    float threadSumGrad = 0.0f;
    float threadSumGradXhat = 0.0f;

    for (int idx = tid; idx < M; idx += blockDim.x) {
        int n = idx / hw;
        int i = idx % hw;
        int inIdx = (n * C + c) * hw + i;
        float go = gradOut[inIdx];
        float xhat = (in[inIdx] - mean) * invVar;
        threadSumGrad += go;
        threadSumGradXhat += go * xhat;
    }

    s_sumGrad[tid] = threadSumGrad;
    s_sumGradXhat[tid] = threadSumGradXhat;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sumGrad[tid] += s_sumGrad[tid + stride];
            s_sumGradXhat[tid] += s_sumGradXhat[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (gradBeta) atomicAdd(&gradBeta[c], s_sumGrad[0]);
        if (gradGamma) atomicAdd(&gradGamma[c], s_sumGradXhat[0]);
    }
    __syncthreads();

    float sumGrad = s_sumGrad[0];
    float sumGradXhat = s_sumGradXhat[0];
    float factor = (g * invVar) / static_cast<float>(M);

    if (gradIn) {
        for (int idx = tid; idx < M; idx += blockDim.x) {
            int n = idx / hw;
            int i = idx % hw;
            int inIdx = (n * C + c) * hw + i;
            float go = gradOut[inIdx];
            float xhat = (in[inIdx] - mean) * invVar;
            gradIn[inIdx] = factor * (static_cast<float>(M) * go - sumGrad - xhat * sumGradXhat);
        }
    }
}

__global__ void k_bce_with_logits_forward(const float* logits, const float* targets, float* lossPerElem, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float z = logits[idx];
        float y = targets[idx];
        float term1 = (z > 0.0f) ? z : 0.0f;
        float term2 = z * y;
        float term3 = log1pf(expf(-fabsf(z)));
        lossPerElem[idx] = term1 - term2 + term3;
    }
}

__global__ void k_bce_with_logits_backward(const float* logits, const float* targets, float* gradLogits, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float z = logits[idx];
        float y = targets[idx];
        float p;
        if (z >= 0.0f) {
            p = 1.0f / (1.0f + expf(-z));
        } else {
            float expZ = expf(z);
            p = expZ / (1.0f + expZ);
        }
        gradLogits[idx] = (p - y) / static_cast<float>(n);
    }
}

__global__ void k_sgd_update(float* weight, const float* grad, float* velocity,
                             size_t n, float lr, float momentum, float weightDecay) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        if (weightDecay != 0.0f) {
            g += weightDecay * weight[idx];
        }
        float v = velocity ? velocity[idx] : 0.0f;
        v = momentum * v + g;
        if (velocity) {
            velocity[idx] = v;
        }
        weight[idx] -= lr * v;
    }
}
#endif // USE_CUDA

// -------------------------------------------------------------
// CPU Implementations
// -------------------------------------------------------------

static void reluForwardCPU(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

static void reluBackwardCPU(const float* in, const float* gradOut, float* gradIn, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        gradIn[i] = (in[i] > 0.0f) ? gradOut[i] : 0.0f;
    }
}

static void maxpool2dForwardCPU(const float* in, float* out, int* mask,
                                int N, int C, int H, int W,
                                int poolH, int poolW, int strideH, int strideW) {
    int outH = (H - poolH) / strideH + 1;
    int outW = (W - poolW) / strideW + 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* inChannel = in + (n * C + c) * H * W;
            float* outChannel = out + (n * C + c) * outH * outW;
            int* maskChannel = mask ? (mask + (n * C + c) * outH * outW) : nullptr;

            for (int oh = 0; oh < outH; ++oh) {
                int hStart = oh * strideH;
                for (int ow = 0; ow < outW; ++ow) {
                    int wStart = ow * strideW;
                    float maxVal = -1e30f;
                    int maxIdx = -1;

                    for (int ph = 0; ph < poolH; ++ph) {
                        int ih = hStart + ph;
                        if (ih >= H) continue;
                        for (int pw = 0; pw < poolW; ++pw) {
                            int iw = wStart + pw;
                            if (iw >= W) continue;
                            int inIdx = ih * W + iw;
                            float val = inChannel[inIdx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = inIdx;
                            }
                        }
                    }

                    int outIdx = oh * outW + ow;
                    outChannel[outIdx] = maxVal;
                    if (maskChannel) {
                        maskChannel[outIdx] = maxIdx;
                    }
                }
            }
        }
    }
}

static void maxpool2dBackwardCPU(const float* gradOut, const int* mask, float* gradIn,
                                 int N, int C, int H, int W,
                                 int outH, int outW) {
    std::fill(gradIn, gradIn + (N * C * H * W), 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float* gradInChannel = gradIn + (n * C + c) * H * W;
            const float* gradOutChannel = gradOut + (n * C + c) * outH * outW;
            const int* maskChannel = mask + (n * C + c) * outH * outW;

            for (int idx = 0; idx < outH * outW; ++idx) {
                int maxIdx = maskChannel[idx];
                if (maxIdx >= 0 && maxIdx < H * W) {
                    gradInChannel[maxIdx] += gradOutChannel[idx];
                }
            }
        }
    }
}

static void globalAvgPoolForwardCPU(const float* in, float* out,
                                    int N, int C, int H, int W) {
    int hw = H * W;
    float scale = 1.0f / static_cast<float>(hw);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* inChannel = in + (n * C + c) * hw;
            float sum = 0.0f;
            for (int i = 0; i < hw; ++i) {
                sum += inChannel[i];
            }
            out[n * C + c] = sum * scale;
        }
    }
}

static void globalAvgPoolBackwardCPU(const float* gradOut, float* gradIn,
                                     int N, int C, int H, int W) {
    int hw = H * W;
    float scale = 1.0f / static_cast<float>(hw);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float val = gradOut[n * C + c] * scale;
            float* inChannel = gradIn + (n * C + c) * hw;
            for (int i = 0; i < hw; ++i) {
                inChannel[i] = val;
            }
        }
    }
}

static void conv2dForwardCPU(const float* in, const float* weight, const float* bias, float* out,
                             int N, int inC, int inH, int inW,
                             int outC, int kH, int kW,
                             int strideH, int strideW, int padH, int padW) {
    int outH = (inH + 2 * padH - kH) / strideH + 1;
    int outW = (inW + 2 * padW - kW) / strideW + 1;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < outC; ++oc) {
            float b = (bias != nullptr) ? bias[oc] : 0.0f;
            float* outChannel = out + ((n * outC + oc) * outH * outW);

            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    float sum = b;
                    for (int ic = 0; ic < inC; ++ic) {
                        const float* inChannel = in + ((n * inC + ic) * inH * inW);
                        const float* kChannel = weight + (((oc * inC + ic) * kH) * kW);

                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = oh * strideH - padH + kh;
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int iw = ow * strideW - padW + kw;
                                if (iw < 0 || iw >= inW) continue;
                                sum += inChannel[ih * inW + iw] * kChannel[kh * kW + kw];
                            }
                        }
                    }
                    outChannel[oh * outW + ow] = sum;
                }
            }
        }
    }
}

static void conv2dBackwardCPU(const float* in, const float* weight, const float* gradOut,
                              float* gradIn, float* gradWeight, float* gradBias,
                              int N, int inC, int inH, int inW,
                              int outC, int kH, int kW,
                              int strideH, int strideW, int padH, int padW) {
    int outH = (inH + 2 * padH - kH) / strideH + 1;
    int outW = (inW + 2 * padW - kW) / strideW + 1;

    if (gradIn) {
        std::fill(gradIn, gradIn + (N * inC * inH * inW), 0.0f);
    }
    if (gradWeight) {
        std::fill(gradWeight, gradWeight + (outC * inC * kH * kW), 0.0f);
    }
    if (gradBias) {
        std::fill(gradBias, gradBias + outC, 0.0f);
    }

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < outC; ++oc) {
            const float* goChannel = gradOut + ((n * outC + oc) * outH * outW);

            if (gradBias) {
                float bSum = 0.0f;
                for (int i = 0; i < outH * outW; ++i) {
                    bSum += goChannel[i];
                }
                gradBias[oc] += bSum;
            }

            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    float go = goChannel[oh * outW + ow];
                    if (go == 0.0f) continue;

                    for (int ic = 0; ic < inC; ++ic) {
                        const float* inChannel = in + ((n * inC + ic) * inH * inW);
                        const float* kChannel = weight + (((oc * inC + ic) * kH) * kW);
                        float* gwChannel = gradWeight ? (gradWeight + (((oc * inC + ic) * kH) * kW)) : nullptr;
                        float* giChannel = gradIn ? (gradIn + ((n * inC + ic) * inH * inW)) : nullptr;

                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = oh * strideH - padH + kh;
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int iw = ow * strideW - padW + kw;
                                if (iw < 0 || iw >= inW) continue;

                                if (gwChannel) {
                                    gwChannel[kh * kW + kw] += go * inChannel[ih * inW + iw];
                                }
                                if (giChannel) {
                                    giChannel[ih * inW + iw] += go * kChannel[kh * kW + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void batchNorm2dForwardCPU(const float* in, float* out,
                                  const float* gamma, const float* beta,
                                  float* runningMean, float* runningVar,
                                  float* saveMean, float* saveInvVar,
                                  int N, int C, int H, int W,
                                  float eps, float momentum, bool training) {
    int hw = H * W;
    int M = N * hw;
    float scale = 1.0f / static_cast<float>(M);

    for (int c = 0; c < C; ++c) {
        float mean = 0.0f;
        float invVar = 1.0f;

        if (training) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                const float* channel = in + (n * C + c) * hw;
                for (int i = 0; i < hw; ++i) {
                    sum += channel[i];
                }
            }
            mean = sum * scale;

            float varSum = 0.0f;
            for (int n = 0; n < N; ++n) {
                const float* channel = in + (n * C + c) * hw;
                for (int i = 0; i < hw; ++i) {
                    float diff = channel[i] - mean;
                    varSum += diff * diff;
                }
            }
            float var = varSum * scale;
            invVar = 1.0f / std::sqrt(var + eps);

            if (saveMean) saveMean[c] = mean;
            if (saveInvVar) saveInvVar[c] = invVar;

            if (runningMean) runningMean[c] = (1.0f - momentum) * runningMean[c] + momentum * mean;
            if (runningVar) runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * var;
        } else {
            mean = runningMean ? runningMean[c] : 0.0f;
            float var = runningVar ? runningVar[c] : 1.0f;
            invVar = 1.0f / std::sqrt(var + eps);
        }

        float g = gamma ? gamma[c] : 1.0f;
        float b = beta ? beta[c] : 0.0f;

        for (int n = 0; n < N; ++n) {
            const float* inChannel = in + (n * C + c) * hw;
            float* outChannel = out + (n * C + c) * hw;
            for (int i = 0; i < hw; ++i) {
                outChannel[i] = g * (inChannel[i] - mean) * invVar + b;
            }
        }
    }
}

static void batchNorm2dBackwardCPU(const float* gradOut, const float* in, float* gradIn,
                                   const float* gamma, float* gradGamma, float* gradBeta,
                                   const float* saveMean, const float* saveInvVar,
                                   int N, int C, int H, int W) {
    int hw = H * W;
    int M = N * hw;

    for (int c = 0; c < C; ++c) {
        float mean = saveMean[c];
        float invVar = saveInvVar[c];
        float g = gamma ? gamma[c] : 1.0f;

        float sumGrad = 0.0f;
        float sumGradXhat = 0.0f;

        for (int n = 0; n < N; ++n) {
            const float* goChan = gradOut + (n * C + c) * hw;
            const float* inChan = in + (n * C + c) * hw;
            for (int i = 0; i < hw; ++i) {
                float go = goChan[i];
                float xhat = (inChan[i] - mean) * invVar;
                sumGrad += go;
                sumGradXhat += go * xhat;
            }
        }

        if (gradBeta) gradBeta[c] += sumGrad;
        if (gradGamma) gradGamma[c] += sumGradXhat;

        if (gradIn) {
            float factor = (g * invVar) / static_cast<float>(M);
            for (int n = 0; n < N; ++n) {
                const float* goChan = gradOut + (n * C + c) * hw;
                const float* inChan = in + (n * C + c) * hw;
                float* giChan = gradIn + (n * C + c) * hw;
                for (int i = 0; i < hw; ++i) {
                    float xhat = (inChan[i] - mean) * invVar;
                    giChan[i] = factor * (static_cast<float>(M) * goChan[i] - sumGrad - xhat * sumGradXhat);
                }
            }
        }
    }
}

static void bceWithLogitsForwardCPU(const float* logits, const float* targets, float* loss, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float z = logits[i];
        float y = targets[i];
        float term1 = (z > 0.0f) ? z : 0.0f;
        float term2 = z * y;
        float term3 = std::log1p(std::exp(-std::abs(z)));
        sum += (term1 - term2 + term3);
    }
    *loss = sum / static_cast<float>(n);
}

static void bceWithLogitsBackwardCPU(const float* logits, const float* targets, float* gradLogits, size_t n) {
    float scale = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) {
        float z = logits[i];
        float y = targets[i];
        float p;
        if (z >= 0.0f) {
            p = 1.0f / (1.0f + std::exp(-z));
        } else {
            float expZ = std::exp(z);
            p = expZ / (1.0f + expZ);
        }
        gradLogits[i] = (p - y) * scale;
    }
}

static void sgdUpdateCPU(float* weight, const float* grad, float* velocity,
                         size_t n, float lr, float momentum, float weightDecay) {
    for (size_t i = 0; i < n; ++i) {
        float g = grad[i];
        if (weightDecay != 0.0f) {
            g += weightDecay * weight[i];
        }
        if (velocity != nullptr && momentum != 0.0f) {
            velocity[i] = momentum * velocity[i] + g;
            weight[i] -= lr * velocity[i];
        } else {
            weight[i] -= lr * g;
        }
    }
}

static void sgemmCPU(bool transA, bool transB,
                     int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float aVal = transA ? A[k * lda + i] : A[i * lda + k];
                float bVal = transB ? B[j * ldb + k] : B[k * ldb + j];
                sum += aVal * bVal;
            }
            float orig = (beta != 0.0f) ? C[i * ldc + j] : 0.0f;
            C[i * ldc + j] = alpha * sum + beta * orig;
        }
    }
}

// -------------------------------------------------------------
// CUDABackend Member Functions
// -------------------------------------------------------------

bool CUDABackend::isAvailable() {
#ifdef USE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

int CUDABackend::getDeviceCount() {
#ifdef USE_CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return 0;
#endif
}

void CUDABackend::setDevice(int deviceId) {
#ifdef USE_CUDA
    CHECK_CUDA(cudaSetDevice(deviceId));
#else
    (void)deviceId;
#endif
}

CUDADeviceInfo CUDABackend::getDeviceInfo(int deviceId) {
    CUDADeviceInfo info;
#ifdef USE_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
        info.deviceId = deviceId;
        info.name = prop.name;
        info.totalGlobalMem = prop.totalGlobalMem;
        info.major = prop.major;
        info.minor = prop.minor;
        info.multiProcessorCount = prop.multiProcessorCount;
        info.isAvailable = true;
    }
#else
    (void)deviceId;
    info.isAvailable = false;
#endif
    return info;
}

float* CUDABackend::allocateDevice(size_t numElements) {
#ifdef USE_CUDA
    float* devPtr = nullptr;
    size_t bytes = numElements * sizeof(float);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&devPtr), bytes));
    return devPtr;
#else
    (void)numElements;
    throw std::runtime_error("CUDA is not compiled/available. Cannot allocate device memory.");
#endif
}

void CUDABackend::freeDevice(float* devPtr) {
#ifdef USE_CUDA
    if (devPtr) {
        cudaFree(devPtr);
    }
#else
    (void)devPtr;
#endif
}

void CUDABackend::copyHostToDevice(float* devDst, const float* hostSrc, size_t numElements) {
#ifdef USE_CUDA
    size_t bytes = numElements * sizeof(float);
    CHECK_CUDA(cudaMemcpy(devDst, hostSrc, bytes, cudaMemcpyHostToDevice));
#else
    (void)devDst; (void)hostSrc; (void)numElements;
    throw std::runtime_error("CUDA is not compiled/available. Cannot copy to device.");
#endif
}

void CUDABackend::copyDeviceToHost(float* hostDst, const float* devSrc, size_t numElements) {
#ifdef USE_CUDA
    size_t bytes = numElements * sizeof(float);
    CHECK_CUDA(cudaMemcpy(hostDst, devSrc, bytes, cudaMemcpyDeviceToHost));
#else
    (void)hostDst; (void)devSrc; (void)numElements;
    throw std::runtime_error("CUDA is not compiled/available. Cannot copy to host.");
#endif
}

void CUDABackend::copyDeviceToDevice(float* devDst, const float* devSrc, size_t numElements) {
#ifdef USE_CUDA
    size_t bytes = numElements * sizeof(float);
    CHECK_CUDA(cudaMemcpy(devDst, devSrc, bytes, cudaMemcpyDeviceToDevice));
#else
    (void)devDst; (void)devSrc; (void)numElements;
    throw std::runtime_error("CUDA is not compiled/available.");
#endif
}

void CUDABackend::setDeviceMemory(float* devPtr, float value, size_t numElements) {
#ifdef USE_CUDA
    if (value == 0.0f) {
        CHECK_CUDA(cudaMemset(devPtr, 0, numElements * sizeof(float)));
    } else {
        int threads = 256;
        int blocks = static_cast<int>((numElements + threads - 1) / threads);
        k_fill<<<blocks, threads>>>(devPtr, value, numElements);
        CHECK_CUDA(cudaGetLastError());
    }
#else
    (void)devPtr; (void)value; (void)numElements;
#endif
}

void CUDABackend::synchronize() {
#ifdef USE_CUDA
    CHECK_CUDA(cudaDeviceSynchronize());
#endif
}

void CUDABackend::addBias2D(float* data, const float* bias, int batchSize, int features, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        for (int b = 0; b < batchSize; ++b) {
            for (int f = 0; f < features; ++f) {
                data[b * features + f] += bias[f];
            }
        }
    } else {
#ifdef USE_CUDA
        int total = batchSize * features;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_add_bias_2d<<<blocks, threads>>>(data, bias, batchSize, features);
        CHECK_CUDA(cudaGetLastError());
#endif
    }
}

void CUDABackend::sumRows(const float* gradOut, float* gradBias, int rows, int cols, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        for (int c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < rows; ++r) {
                sum += gradOut[r * cols + c];
            }
            gradBias[c] += sum;
        }
    } else {
#ifdef USE_CUDA
        int threads = 256;
        int blocks = (cols + threads - 1) / threads;
        k_sum_rows<<<blocks, threads>>>(gradOut, gradBias, rows, cols);
        CHECK_CUDA(cudaGetLastError());
#endif
    }
}

void CUDABackend::accumulate(float* dst, const float* src, size_t n, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] += src[i];
        }
    } else {
#ifdef USE_CUDA
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        k_accumulate<<<blocks, threads>>>(dst, src, n);
        CHECK_CUDA(cudaGetLastError());
#endif
    }
}

void CUDABackend::sgemm(bool transA, bool transB,
                       int M, int N, int K,
                       float alpha,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float beta,
                       float* C, int ldc,
                       DeviceType device) {
    if (device == DeviceType::CPU) {
        sgemmCPU(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#ifdef USE_CUDA
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // For row-major C = A * B:
    // In cuBLAS (col-major): C^T = B^T * A^T
    // We pass B as first matrix and A as second
    cublasOperation_t cublasOpB = (opB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cublasOpA = (opA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasHandle_t handle = getCublasHandle();
    CHECK_CUBLAS(cublasSgemm(handle,
                             cublasOpB, cublasOpA,
                             N, M, K,
                             &alpha,
                             B, ldb,
                             A, lda,
                             &beta,
                             C, ldc));
#else
    sgemmCPU(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void CUDABackend::reluForward(const float* input, float* output, size_t n, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        reluForwardCPU(input, output, n);
    } else {
#ifdef USE_CUDA
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        k_relu_forward<<<blocks, threads>>>(input, output, n);
        CHECK_CUDA(cudaGetLastError());
#else
        reluForwardCPU(input, output, n);
#endif
    }
}

void CUDABackend::reluBackward(const float* input, const float* gradOutput, float* gradInput, size_t n, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        reluBackwardCPU(input, gradOutput, gradInput, n);
    } else {
#ifdef USE_CUDA
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        k_relu_backward<<<blocks, threads>>>(input, gradOutput, gradInput, n);
        CHECK_CUDA(cudaGetLastError());
#else
        reluBackwardCPU(input, gradOutput, gradInput, n);
#endif
    }
}

void CUDABackend::maxpool2dForward(const float* input, float* output, int* mask,
                                  int N, int C, int H, int W,
                                  int poolH, int poolW, int strideH, int strideW,
                                  DeviceType dev) {
    int outH = (H - poolH) / strideH + 1;
    int outW = (W - poolW) / strideW + 1;
    if (dev == DeviceType::CPU) {
        maxpool2dForwardCPU(input, output, mask, N, C, H, W, poolH, poolW, strideH, strideW);
    } else {
#ifdef USE_CUDA
        int total = N * C * outH * outW;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_maxpool2d_forward<<<blocks, threads>>>(input, output, mask, N, C, H, W, outH, outW, poolH, poolW, strideH, strideW);
        CHECK_CUDA(cudaGetLastError());
#else
        maxpool2dForwardCPU(input, output, mask, N, C, H, W, poolH, poolW, strideH, strideW);
#endif
    }
}

void CUDABackend::maxpool2dBackward(const float* gradOutput, const int* mask, float* gradInput,
                                   int N, int C, int H, int W,
                                   int outH, int outW,
                                   DeviceType dev) {
    if (dev == DeviceType::CPU) {
        maxpool2dBackwardCPU(gradOutput, mask, gradInput, N, C, H, W, outH, outW);
    } else {
#ifdef USE_CUDA
        CHECK_CUDA(cudaMemset(gradInput, 0, N * C * H * W * sizeof(float)));
        int total = N * C * outH * outW;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_maxpool2d_backward<<<blocks, threads>>>(gradOutput, mask, gradInput, N, C, H, W, outH, outW);
        CHECK_CUDA(cudaGetLastError());
#else
        maxpool2dBackwardCPU(gradOutput, mask, gradInput, N, C, H, W, outH, outW);
#endif
    }
}

void CUDABackend::globalAvgPoolForward(const float* input, float* output,
                                      int N, int C, int H, int W,
                                      DeviceType dev) {
    if (dev == DeviceType::CPU) {
        globalAvgPoolForwardCPU(input, output, N, C, H, W);
    } else {
#ifdef USE_CUDA
        int total = N * C;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_global_avg_pool_forward<<<blocks, threads>>>(input, output, N, C, H, W);
        CHECK_CUDA(cudaGetLastError());
#else
        globalAvgPoolForwardCPU(input, output, N, C, H, W);
#endif
    }
}

void CUDABackend::globalAvgPoolBackward(const float* gradOutput, float* gradInput,
                                       int N, int C, int H, int W,
                                       DeviceType dev) {
    if (dev == DeviceType::CPU) {
        globalAvgPoolBackwardCPU(gradOutput, gradInput, N, C, H, W);
    } else {
#ifdef USE_CUDA
        int total = N * C * H * W;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_global_avg_pool_backward<<<blocks, threads>>>(gradOutput, gradInput, N, C, H, W);
        CHECK_CUDA(cudaGetLastError());
#else
        globalAvgPoolBackwardCPU(gradOutput, gradInput, N, C, H, W);
#endif
    }
}

void CUDABackend::conv2dForward(const float* input, const float* weight, const float* bias, float* output,
                               int N, int inC, int inH, int inW,
                               int outC, int kH, int kW,
                               int strideH, int strideW, int padH, int padW,
                               DeviceType dev) {
    if (dev == DeviceType::CPU) {
        conv2dForwardCPU(input, weight, bias, output, N, inC, inH, inW, outC, kH, kW, strideH, strideW, padH, padW);
        return;
    }

#ifdef USE_CUDA
    int outH = (inH + 2 * padH - kH) / strideH + 1;
    int outW = (inW + 2 * padW - kW) / strideW + 1;

    int M = outC;
    int K = inC * kH * kW;
    int S = outH * outW;

    float* col_buffer = getColWorkspace(static_cast<size_t>(K) * S);

    int in_image_size = inC * inH * inW;
    int out_image_size = outC * S;

    int threads = 256;
    int im2col_total = inC * S;
    int im2col_blocks = (im2col_total + threads - 1) / threads;

    for (int n = 0; n < N; ++n) {
        const float* in_img = input + n * in_image_size;
        float* out_img = output + n * out_image_size;

        // Step 1: im2col
        k_im2col<<<im2col_blocks, threads>>>(in_img, col_buffer,
                                             inC, inH, inW,
                                             kH, kW,
                                             padH, padW,
                                             strideH, strideW,
                                             outH, outW);

        // Step 2: cuBLAS SGEMM (Output = W * Col)
        // W is [M x K], Col is [K x S], Output is [M x S]
        sgemm(false, false, M, S, K, 1.0f, weight, K, col_buffer, S, 0.0f, out_img, S, DeviceType::CUDA);
    }

    // Step 3: Add bias
    if (bias != nullptr) {
        int total_out = N * outC * S;
        int bias_blocks = (total_out + threads - 1) / threads;
        k_add_bias_channel<<<bias_blocks, threads>>>(output, bias, total_out, outC, S);
    }
    CHECK_CUDA(cudaGetLastError());
#else
    conv2dForwardCPU(input, weight, bias, output, N, inC, inH, inW, outC, kH, kW, strideH, strideW, padH, padW);
#endif
}

void CUDABackend::conv2dBackward(const float* input, const float* weight, const float* gradOutput,
                                float* gradInput, float* gradWeight, float* gradBias,
                                int N, int inC, int inH, int inW,
                                int outC, int kH, int kW,
                                int strideH, int strideW, int padH, int padW,
                                DeviceType dev) {
    if (dev == DeviceType::CPU) {
        conv2dBackwardCPU(input, weight, gradOutput, gradInput, gradWeight, gradBias,
                          N, inC, inH, inW, outC, kH, kW, strideH, strideW, padH, padW);
        return;
    }

#ifdef USE_CUDA
    int outH = (inH + 2 * padH - kH) / strideH + 1;
    int outW = (inW + 2 * padW - kW) / strideW + 1;

    int M = outC;
    int K = inC * kH * kW;
    int S = outH * outW;

    int in_image_size = inC * inH * inW;
    int out_image_size = outC * S;
    int threads = 256;

    // 1. Bias gradient (coalesced reduction kernel: 1 block per channel, 256 threads)
    if (gradBias != nullptr) {
        k_conv2d_bias_backward<<<outC, 256>>>(gradOutput, gradBias, N, outC, S);
    }

    // 2. Weight gradient & Input gradient
    if (gradWeight != nullptr) {
        CHECK_CUDA(cudaMemset(gradWeight, 0, outC * inC * kH * kW * sizeof(float)));
    }
    if (gradInput != nullptr) {
        CHECK_CUDA(cudaMemset(gradInput, 0, N * inC * inH * inW * sizeof(float)));
    }

    if (gradWeight != nullptr || gradInput != nullptr) {
        float* col_buffer = getColWorkspace(static_cast<size_t>(K) * S);

        int im2col_total = inC * S;
        int im2col_blocks = (im2col_total + threads - 1) / threads;

        int col2im_total = inC * inH * inW;
        int col2im_blocks = (col2im_total + threads - 1) / threads;

        for (int n = 0; n < N; ++n) {
            const float* in_img = input + n * in_image_size;
            const float* go_img = gradOutput + n * out_image_size;
            float* gi_img = gradInput ? (gradInput + n * in_image_size) : nullptr;

            // Compute gradWeight: dW += go_img * (col)^T
            if (gradWeight != nullptr) {
                // im2col on input image
                k_im2col<<<im2col_blocks, threads>>>(in_img, col_buffer,
                                                     inC, inH, inW,
                                                     kH, kW,
                                                     padH, padW,
                                                     strideH, strideW,
                                                     outH, outW);

                // cuBLAS SGEMM: dW += go_img * (col)^T
                // go_img is [M x S], col is [K x S]. col^T is [S x K].
                // Result dW is [M x K]. Accumulate across batch with beta = 1.0f!
                sgemm(false, true, M, K, S, 1.0f, go_img, S, col_buffer, S, 1.0f, gradWeight, K, DeviceType::CUDA);
            }

            // Compute gradInput: dCol = W^T * go_img, then col2im
            if (gradInput != nullptr) {
                // cuBLAS SGEMM: dCol = W^T * go_img
                // W is [M x K], W^T is [K x M]. go_img is [M x S].
                // Result dCol is [K x S].
                sgemm(true, false, K, S, M, 1.0f, weight, K, go_img, S, 0.0f, col_buffer, S, DeviceType::CUDA);

                // col2im kernel: folds dCol [K x S] -> gi_img [inC x inH x inW]
                k_col2im<<<col2im_blocks, threads>>>(col_buffer, gi_img,
                                                     inC, inH, inW,
                                                     kH, kW,
                                                     padH, padW,
                                                     strideH, strideW,
                                                     outH, outW);
            }
        }
    }
    CHECK_CUDA(cudaGetLastError());
#else
    conv2dBackwardCPU(input, weight, gradOutput, gradInput, gradWeight, gradBias,
                      N, inC, inH, inW, outC, kH, kW, strideH, strideW, padH, padW);
#endif
}

void CUDABackend::batchNorm2dForward(const float* in, float* out,
                                    const float* gamma, const float* beta,
                                    float* runningMean, float* runningVar,
                                    float* saveMean, float* saveInvVar,
                                    int N, int C, int H, int W,
                                    float eps, float momentum, bool training,
                                    DeviceType dev) {
    if (dev == DeviceType::CPU) {
        batchNorm2dForwardCPU(in, out, gamma, beta, runningMean, runningVar, saveMean, saveInvVar,
                              N, C, H, W, eps, momentum, training);
    } else {
#ifdef USE_CUDA
        if (training) {
            k_batchnorm2d_forward_train<<<C, 256>>>(in, out, gamma, beta, runningMean, runningVar,
                                                    saveMean, saveInvVar, N, C, H, W, eps, momentum);
            CHECK_CUDA(cudaGetLastError());
        } else {
            int total = N * C * H * W;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            k_batchnorm2d_forward_eval<<<blocks, threads>>>(in, out, gamma, beta, runningMean, runningVar,
                                                            total, C, H, W, eps);
            CHECK_CUDA(cudaGetLastError());
        }
#else
        batchNorm2dForwardCPU(in, out, gamma, beta, runningMean, runningVar, saveMean, saveInvVar,
                              N, C, H, W, eps, momentum, training);
#endif
    }
}

void CUDABackend::batchNorm2dBackward(const float* gradOut, const float* in, float* gradIn,
                                     const float* gamma, float* gradGamma, float* gradBeta,
                                     const float* saveMean, const float* saveInvVar,
                                     int N, int C, int H, int W,
                                     DeviceType dev) {
    if (dev == DeviceType::CPU) {
        batchNorm2dBackwardCPU(gradOut, in, gradIn, gamma, gradGamma, gradBeta, saveMean, saveInvVar,
                               N, C, H, W);
    } else {
#ifdef USE_CUDA
        k_batchnorm2d_backward<<<C, 256>>>(gradOut, in, gradIn, gamma, gradGamma, gradBeta,
                                           saveMean, saveInvVar, N, C, H, W);
        CHECK_CUDA(cudaGetLastError());
#else
        batchNorm2dBackwardCPU(gradOut, in, gradIn, gamma, gradGamma, gradBeta, saveMean, saveInvVar,
                               N, C, H, W);
#endif
    }
}

void CUDABackend::bceWithLogitsForward(const float* logits, const float* targets, float* loss,
                                      size_t n, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        bceWithLogitsForwardCPU(logits, targets, loss, n);
    } else {
#ifdef USE_CUDA
        float* d_lossElem = nullptr;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_lossElem), n * sizeof(float)));
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        k_bce_with_logits_forward<<<blocks, threads>>>(logits, targets, d_lossElem, n);
        CHECK_CUDA(cudaGetLastError());

        std::vector<float> h_elem(n);
        CHECK_CUDA(cudaMemcpy(h_elem.data(), d_lossElem, n * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_lossElem);

        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += h_elem[i];
        }
        *loss = sum / static_cast<float>(n);
#else
        bceWithLogitsForwardCPU(logits, targets, loss, n);
#endif
    }
}

void CUDABackend::bceWithLogitsBackward(const float* logits, const float* targets, float* gradLogits,
                                       size_t n, DeviceType dev) {
    if (dev == DeviceType::CPU) {
        bceWithLogitsBackwardCPU(logits, targets, gradLogits, n);
    } else {
#ifdef USE_CUDA
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        k_bce_with_logits_backward<<<blocks, threads>>>(logits, targets, gradLogits, n);
        CHECK_CUDA(cudaGetLastError());
#else
        bceWithLogitsBackwardCPU(logits, targets, gradLogits, n);
#endif
    }
}

void CUDABackend::sgdUpdate(float* weight, const float* grad, float* velocity,
                           size_t n, float lr, float momentum, float weightDecay,
                           DeviceType dev) {
    if (dev == DeviceType::CPU) {
        sgdUpdateCPU(weight, grad, velocity, n, lr, momentum, weightDecay);
    } else {
#ifdef USE_CUDA
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        k_sgd_update<<<blocks, threads>>>(weight, grad, velocity, n, lr, momentum, weightDecay);
        CHECK_CUDA(cudaGetLastError());
#else
        sgdUpdateCPU(weight, grad, velocity, n, lr, momentum, weightDecay);
#endif
    }
}

void CUDABackend::checkCuda(int status, const char* file, int line) {
#ifdef USE_CUDA
    if (status != cudaSuccess) {
        std::string msg = "CUDA Error at " + std::string(file) + ":" + std::to_string(line) + " - " + cudaGetErrorString((cudaError_t)status);
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }
#else
    (void)status; (void)file; (void)line;
#endif
}

void CUDABackend::checkCublas(int status, const char* file, int line) {
#ifdef USE_CUDA
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::string msg = "cuBLAS Error at " + std::string(file) + ":" + std::to_string(line) + " (code: " + std::to_string(status) + ")";
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }
#else
    (void)status; (void)file; (void)line;
#endif
}
