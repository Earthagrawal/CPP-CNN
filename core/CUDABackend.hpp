#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <iostream>

// CUDA Error Checking Macro
#define CHECK_CUDA(call) \
    do { \
        CUDABackend::checkCuda((call), __FILE__, __LINE__); \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        CUDABackend::checkCublas((call), __FILE__, __LINE__); \
    } while (0)

enum class DeviceType {
    CPU = 0,
    CUDA = 1
};

// Device properties summary
struct CUDADeviceInfo {
    int deviceId = -1;
    std::string name;
    size_t totalGlobalMem = 0;
    int major = 0;
    int minor = 0;
    int multiProcessorCount = 0;
    bool isAvailable = false;
};

// CUDABackend: Central interface for GPU memory, device info, cuBLAS, and kernel execution
class CUDABackend {
public:
    // Device management
    static bool isAvailable();
    static int getDeviceCount();
    static void setDevice(int deviceId);
    static CUDADeviceInfo getDeviceInfo(int deviceId = 0);

    // Memory management
    static float* allocateDevice(size_t numElements);
    static void freeDevice(float* devPtr);

    // Data transfers
    static void copyHostToDevice(float* devDst, const float* hostSrc, size_t numElements);
    static void copyDeviceToHost(float* hostDst, const float* devSrc, size_t numElements);
    static void copyDeviceToDevice(float* devDst, const float* devSrc, size_t numElements);
    static void setDeviceMemory(float* devPtr, float value, size_t numElements);
    static void synchronize();

    // cuBLAS Matrix Multiplication: C = alpha * (opA(A)) * (opB(B)) + beta * C
    // In row-major format: A is [M x K], B is [K x N], C is [M x N]
    static void sgemm(bool transA, bool transB,
                      int M, int N, int K,
                      float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta,
                      float* C, int ldc,
                      DeviceType device = DeviceType::CPU);

    // Kernels and activations
    static void reluForward(const float* input, float* output, size_t n, DeviceType dev);
    static void reluBackward(const float* input, const float* gradOutput, float* gradInput, size_t n, DeviceType dev);

    static void maxpool2dForward(const float* input, float* output, int* mask,
                                 int N, int C, int H, int W,
                                 int poolH, int poolW, int strideH, int strideW,
                                 DeviceType dev);
    static void maxpool2dBackward(const float* gradOutput, const int* mask, float* gradInput,
                                  int N, int C, int H, int W,
                                  int outH, int outW,
                                  DeviceType dev);

    static void globalAvgPoolForward(const float* input, float* output,
                                     int N, int C, int H, int W,
                                     DeviceType dev);
    static void globalAvgPoolBackward(const float* gradOutput, float* gradInput,
                                      int N, int C, int H, int W,
                                      DeviceType dev);

    static void conv2dForward(const float* input, const float* weight, const float* bias, float* output,
                              int N, int inC, int inH, int inW,
                              int outC, int kH, int kW,
                              int strideH, int strideW, int padH, int padW,
                              DeviceType dev);
    static void conv2dBackward(const float* input, const float* weight, const float* gradOutput,
                               float* gradInput, float* gradWeight, float* gradBias,
                               int N, int inC, int inH, int inW,
                               int outC, int kH, int kW,
                               int strideH, int strideW, int padH, int padW,
                               DeviceType dev);

    static void batchNorm2dForward(const float* in, float* out,
                                   const float* gamma, const float* beta,
                                   float* runningMean, float* runningVar,
                                   float* saveMean, float* saveInvVar,
                                   int N, int C, int H, int W,
                                   float eps, float momentum, bool training,
                                   DeviceType dev);
    static void batchNorm2dBackward(const float* gradOut, const float* in, float* gradIn,
                                    const float* gamma, float* gradGamma, float* gradBeta,
                                    const float* saveMean, const float* saveInvVar,
                                    int N, int C, int H, int W,
                                    DeviceType dev);

    static void bceWithLogitsForward(const float* logits, const float* targets, float* loss,
                                     size_t n, DeviceType dev);
    static void bceWithLogitsBackward(const float* logits, const float* targets, float* gradLogits,
                                      size_t n, DeviceType dev);

    static void sgdUpdate(float* weight, const float* grad, float* velocity,
                          size_t n, float lr, float momentum, float weightDecay,
                          DeviceType dev);

    // Linear / general tensor elementwise helpers
    static void addBias2D(float* data, const float* bias, int batchSize, int features, DeviceType dev);
    static void sumRows(const float* gradOut, float* gradBias, int rows, int cols, DeviceType dev);
    static void accumulate(float* dst, const float* src, size_t n, DeviceType dev);

    // Internal error check helpers
    static void checkCuda(int status, const char* file, int line);
    static void checkCublas(int status, const char* file, int line);
};
