#include "../core/CUDABackend.hpp"
#include "../core/Tensor.hpp"
#include "../core/Matrix.hpp"
#include "../core/Parameter.hpp"
#include "../layers/Linear.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/MaxPool2D.hpp"
#include "../layers/GlobalAveragePool2D.hpp"
#include "../loss/BCEWithLogitsLoss.hpp"
#include "../optimizers/SGD.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

#define GPU_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] GPU TEST FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

static float maxDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9f;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void printMemInfo(const std::string& label) {
#ifdef USE_CUDA
    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    double usedMB = static_cast<double>(totalBytes - freeBytes) / (1024.0 * 1024.0);
    double totalMB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);
    std::cout << "[VRAM " << label << "] Allocated: " << std::fixed << std::setprecision(1)
              << usedMB << " MB / " << totalMB << " MB" << std::endl;
#else
    (void)label;
#endif
}

// -------------------------------------------------------------
// Test 1: Device Query & Detection
// -------------------------------------------------------------
bool test_device_query() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 1: Hardware Detection & Device Query" << std::endl;
    std::cout << "==========================================" << std::endl;

    GPU_ASSERT(CUDABackend::isAvailable(), "CUDABackend::isAvailable() must be TRUE");
    int count = CUDABackend::getDeviceCount();
    GPU_ASSERT(count > 0, "At least 1 CUDA device must be detected");

    CUDADeviceInfo info = CUDABackend::getDeviceInfo(0);
    std::cout << "[+] Device Found: " << info.name << std::endl;
    std::cout << "  - Compute Capability: " << info.major << "." << info.minor << std::endl;
    std::cout << "  - Total Global VRAM: " << (info.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  - Multiprocessors (SMs): " << info.multiProcessorCount << std::endl;

    GPU_ASSERT(info.totalGlobalMem > 0, "GPU Memory must be > 0");
    printMemInfo("Initial");
    return true;
}

// -------------------------------------------------------------
// Test 2: CUDA Memory Allocation & Transfers
// -------------------------------------------------------------
bool test_memory_transfers() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 2: Memory Allocation, H2D/D2H Transfers & Fill" << std::endl;
    std::cout << "==========================================" << std::endl;

    size_t n = 1024 * 1024; // 1M floats = 4 MB
    std::vector<float> h_src(n);
    for (size_t i = 0; i < n; ++i) h_src[i] = static_cast<float>(i % 1000) * 0.5f;

    Tensor t_gpu({n}, DeviceType::CUDA);
    t_gpu.copyFromHost(h_src.data(), n);

    std::vector<float> h_dst = t_gpu.toHostVector();
    float diff = maxDiff(h_src, h_dst);
    std::cout << "  - Host-to-Device -> Device-to-Host Max Diff: " << diff << std::endl;
    GPU_ASSERT(diff == 0.0f, "H2D and D2H transfers must be exact bitwise");

    // Test GPU fill
    t_gpu.fill(3.14159f);
    h_dst = t_gpu.toHostVector();
    float fillDiff = 0.0f;
    for (float v : h_dst) {
        float d = std::abs(v - 3.14159f);
        if (d > fillDiff) fillDiff = d;
    }
    std::cout << "  - GPU In-Place Fill Max Diff: " << fillDiff << std::endl;
    GPU_ASSERT(fillDiff < 1e-5f, "GPU fill must match target constant");

    std::cout << "[+] CUDA Memory transfers verified successfully." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 3: cuBLAS SGEMM vs CPU Reference
// -------------------------------------------------------------
bool test_cublas_sgemm() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 3: cuBLAS SGEMM vs CPU Reference" << std::endl;
    std::cout << "==========================================" << std::endl;

    int M = 64, K = 128, N = 32;
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>((i % 23) - 11) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>((i % 19) - 9) * 0.1f;

    // CPU Multiply
    Matrix A_cpu(M, K, h_A, DeviceType::CPU);
    Matrix B_cpu(K, N, h_B, DeviceType::CPU);
    Matrix C_cpu = Matrix::matmul(A_cpu, B_cpu);

    // GPU cuBLAS Multiply
    Matrix A_gpu(M, K, h_A, DeviceType::CUDA);
    Matrix B_gpu(K, N, h_B, DeviceType::CUDA);
    Matrix C_gpu = Matrix::matmul(A_gpu, B_gpu);

    std::vector<float> res_cpu = C_cpu.tensor().toHostVector();
    std::vector<float> res_gpu = C_gpu.tensor().toHostVector();
    float diff = maxDiff(res_cpu, res_gpu);
    std::cout << "  - cuBLAS (M=" << M << ", K=" << K << ", N=" << N << ") Max Diff vs CPU: " << diff << std::endl;
    GPU_ASSERT(diff < 1e-4f, "cuBLAS output must match CPU within 1e-4 tolerance");

    // Transposed test: C = A * B^T where B is [N x K]
    std::vector<float> h_BT(N * K);
    for (int i = 0; i < N * K; ++i) h_BT[i] = static_cast<float>((i % 17) - 8) * 0.1f;
    Matrix B_trans_cpu(N, K, h_BT, DeviceType::CPU);
    Matrix B_trans_gpu(N, K, h_BT, DeviceType::CUDA);

    Matrix C_t_cpu = Matrix::matmul(A_cpu, B_trans_cpu, false, true);
    Matrix C_t_gpu = Matrix::matmul(A_gpu, B_trans_gpu, false, true);
    float diffT = maxDiff(C_t_cpu.tensor().toHostVector(), C_t_gpu.tensor().toHostVector());
    std::cout << "  - cuBLAS Transposed B Max Diff vs CPU: " << diffT << std::endl;
    GPU_ASSERT(diffT < 1e-4f, "cuBLAS transposed GEMM must match CPU within 1e-4 tolerance");

    std::cout << "[+] cuBLAS SGEMM verified successfully." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 4: Linear Layer (Forward & Backward)
// -------------------------------------------------------------
bool test_linear_layer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 4: Linear Layer GPU vs CPU (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    size_t B = 4, inFeats = 64, outFeats = 32;
    Linear fc_cpu(inFeats, outFeats, true, DeviceType::CPU);
    Linear fc_gpu(inFeats, outFeats, true, DeviceType::CUDA);

    // Sync weights & biases
    fc_gpu.weight().data().copyFrom(fc_cpu.weight().data());
    fc_gpu.bias().data().copyFrom(fc_cpu.bias().data());

    std::vector<float> h_in(B * inFeats);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.1f;

    Tensor in_cpu({B, inFeats}, h_in, DeviceType::CPU);
    Tensor in_gpu({B, inFeats}, h_in, DeviceType::CUDA);

    Tensor out_cpu = fc_cpu.forward(in_cpu);
    Tensor out_gpu = fc_gpu.forward(in_gpu);

    std::vector<float> h_out_cpu = out_cpu.toHostVector();
    std::vector<float> h_out_gpu = out_gpu.toHostVector();
    float fwdDiff = maxDiff(h_out_cpu, h_out_gpu);
    std::cout << "  - Linear Forward Output Max Diff: " << fwdDiff << std::endl;
    GPU_ASSERT(fwdDiff < 1e-4f, "Linear forward GPU must match CPU");

    // Backward pass
    std::vector<float> h_go(B * outFeats);
    for (size_t i = 0; i < h_go.size(); ++i) h_go[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.1f;

    Tensor go_cpu({B, outFeats}, h_go, DeviceType::CPU);
    Tensor go_gpu({B, outFeats}, h_go, DeviceType::CUDA);

    fc_cpu.weight().zeroGrad(); fc_cpu.bias().zeroGrad();
    fc_gpu.weight().zeroGrad(); fc_gpu.bias().zeroGrad();

    Tensor gi_cpu = fc_cpu.backward(go_cpu);
    Tensor gi_gpu = fc_gpu.backward(go_gpu);

    float giDiff = maxDiff(gi_cpu.toHostVector(), gi_gpu.toHostVector());
    float gwDiff = maxDiff(fc_cpu.weight().grad().toHostVector(), fc_gpu.weight().grad().toHostVector());
    float gbDiff = maxDiff(fc_cpu.bias().grad().toHostVector(), fc_gpu.bias().grad().toHostVector());

    std::cout << "  - Backward gradInput Max Diff: " << giDiff << std::endl;
    std::cout << "  - Backward gradWeight Max Diff: " << gwDiff << std::endl;
    std::cout << "  - Backward gradBias Max Diff: " << gbDiff << std::endl;

    GPU_ASSERT(giDiff < 1e-4f, "Linear gradInput must match CPU");
    GPU_ASSERT(gwDiff < 1e-4f, "Linear gradWeight must match CPU");
    GPU_ASSERT(gbDiff < 1e-4f, "Linear gradBias must match CPU");

    std::cout << "[+] Linear layer verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 5: Conv2D Layer (Forward & Backward)
// -------------------------------------------------------------
bool test_conv2d_layer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 5: Conv2D Layer GPU vs CPU (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    size_t N = 2, inC = 3, inH = 14, inW = 14, outC = 8, k = 3, stride = 1, pad = 1;
    Conv2D conv_cpu(inC, outC, k, stride, pad, true, DeviceType::CPU);
    Conv2D conv_gpu(inC, outC, k, stride, pad, true, DeviceType::CUDA);

    conv_gpu.weight().data().copyFrom(conv_cpu.weight().data());
    conv_gpu.bias().data().copyFrom(conv_cpu.bias().data());

    std::vector<float> h_in(N * inC * inH * inW);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(static_cast<int>(i % 37) - 18) * 0.05f;

    Tensor in_cpu({N, inC, inH, inW}, h_in, DeviceType::CPU);
    Tensor in_gpu({N, inC, inH, inW}, h_in, DeviceType::CUDA);

    Tensor out_cpu = conv_cpu.forward(in_cpu);
    Tensor out_gpu = conv_gpu.forward(in_gpu);

    float fwdDiff = maxDiff(out_cpu.toHostVector(), out_gpu.toHostVector());
    std::cout << "  - Conv2D Forward Output Max Diff: " << fwdDiff << std::endl;
    GPU_ASSERT(fwdDiff < 1e-4f, "Conv2D forward GPU must match CPU");

    // Backward pass
    size_t outH = (inH + 2 * pad - k) / stride + 1;
    size_t outW = (inW + 2 * pad - k) / stride + 1;
    std::vector<float> h_go(N * outC * outH * outW);
    for (size_t i = 0; i < h_go.size(); ++i) h_go[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.1f;

    Tensor go_cpu({N, outC, outH, outW}, h_go, DeviceType::CPU);
    Tensor go_gpu({N, outC, outH, outW}, h_go, DeviceType::CUDA);

    conv_cpu.weight().zeroGrad(); conv_cpu.bias().zeroGrad();
    conv_gpu.weight().zeroGrad(); conv_gpu.bias().zeroGrad();

    Tensor gi_cpu = conv_cpu.backward(go_cpu);
    Tensor gi_gpu = conv_gpu.backward(go_gpu);

    float giDiff = maxDiff(gi_cpu.toHostVector(), gi_gpu.toHostVector());
    float gwDiff = maxDiff(conv_cpu.weight().grad().toHostVector(), conv_gpu.weight().grad().toHostVector());
    float gbDiff = maxDiff(conv_cpu.bias().grad().toHostVector(), conv_gpu.bias().grad().toHostVector());

    std::cout << "  - Conv2D gradInput Max Diff: " << giDiff << std::endl;
    std::cout << "  - Conv2D gradWeight Max Diff: " << gwDiff << std::endl;
    std::cout << "  - Conv2D gradBias Max Diff: " << gbDiff << std::endl;

    GPU_ASSERT(giDiff < 1e-4f, "Conv2D gradInput must match CPU");
    GPU_ASSERT(gwDiff < 1e-4f, "Conv2D gradWeight must match CPU");
    GPU_ASSERT(gbDiff < 1e-4f, "Conv2D gradBias must match CPU");

    std::cout << "[+] Conv2D layer verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 6: ReLU Layer (Forward & Backward)
// -------------------------------------------------------------
bool test_relu_layer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 6: ReLU Layer GPU vs CPU (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    ReLU relu_cpu, relu_gpu;
    std::vector<float> h_in = {-2.5f, 0.0f, 3.8f, -0.1f, 10.2f, -15.0f};
    Tensor in_cpu({h_in.size()}, h_in, DeviceType::CPU);
    Tensor in_gpu({h_in.size()}, h_in, DeviceType::CUDA);

    Tensor out_cpu = relu_cpu.forward(in_cpu);
    Tensor out_gpu = relu_gpu.forward(in_gpu);
    float fwdDiff = maxDiff(out_cpu.toHostVector(), out_gpu.toHostVector());
    std::cout << "  - ReLU Forward Output Max Diff: " << fwdDiff << std::endl;
    GPU_ASSERT(fwdDiff == 0.0f, "ReLU forward must be exact");

    Tensor go({h_in.size()}, 1.0f);
    Tensor gi_cpu = relu_cpu.backward(go.to(DeviceType::CPU));
    Tensor gi_gpu = relu_gpu.backward(go.to(DeviceType::CUDA));
    float bwdDiff = maxDiff(gi_cpu.toHostVector(), gi_gpu.toHostVector());
    std::cout << "  - ReLU Backward GradInput Max Diff: " << bwdDiff << std::endl;
    GPU_ASSERT(bwdDiff == 0.0f, "ReLU backward must be exact");

    std::cout << "[+] ReLU layer verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 7: MaxPool2D Layer (Forward & Backward)
// -------------------------------------------------------------
bool test_maxpool_layer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 7: MaxPool2D Layer GPU vs CPU (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    MaxPool2D pool_cpu(2, 2);
    MaxPool2D pool_gpu(2, 2);

    size_t N = 2, C = 4, H = 8, W = 8;
    std::vector<float> h_in(N * C * H * W);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(static_cast<int>(i % 41) - 20) * 0.1f;

    Tensor in_cpu({N, C, H, W}, h_in, DeviceType::CPU);
    Tensor in_gpu({N, C, H, W}, h_in, DeviceType::CUDA);

    Tensor out_cpu = pool_cpu.forward(in_cpu);
    Tensor out_gpu = pool_gpu.forward(in_gpu);

    float fwdDiff = maxDiff(out_cpu.toHostVector(), out_gpu.toHostVector());
    std::cout << "  - MaxPool2D Forward Output Max Diff: " << fwdDiff << std::endl;
    GPU_ASSERT(fwdDiff == 0.0f, "MaxPool forward must be exact");

    size_t outSize = out_cpu.size();
    std::vector<float> h_go(outSize);
    for (size_t i = 0; i < outSize; ++i) h_go[i] = static_cast<float>(i + 1) * 0.1f;

    Tensor go_cpu(out_cpu.shape(), h_go, DeviceType::CPU);
    Tensor go_gpu(out_gpu.shape(), h_go, DeviceType::CUDA);

    Tensor gi_cpu = pool_cpu.backward(go_cpu);
    Tensor gi_gpu = pool_gpu.backward(go_gpu);

    float bwdDiff = maxDiff(gi_cpu.toHostVector(), gi_gpu.toHostVector());
    std::cout << "  - MaxPool2D Backward GradInput Max Diff: " << bwdDiff << std::endl;
    GPU_ASSERT(bwdDiff == 0.0f, "MaxPool backward argmax routing must be exact");

    std::cout << "[+] MaxPool2D layer verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 8: GlobalAveragePool2D Layer (Forward & Backward)
// -------------------------------------------------------------
bool test_gap_layer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 8: GlobalAveragePool2D Layer GPU vs CPU (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    GlobalAveragePool2D gap_cpu, gap_gpu;
    size_t N = 2, C = 16, H = 7, W = 7;
    std::vector<float> h_in(N * C * H * W);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(static_cast<int>(i % 53) - 26) * 0.1f;

    Tensor in_cpu({N, C, H, W}, h_in, DeviceType::CPU);
    Tensor in_gpu({N, C, H, W}, h_in, DeviceType::CUDA);

    Tensor out_cpu = gap_cpu.forward(in_cpu);
    Tensor out_gpu = gap_gpu.forward(in_gpu);

    float fwdDiff = maxDiff(out_cpu.toHostVector(), out_gpu.toHostVector());
    std::cout << "  - GAP Forward Output Max Diff: " << fwdDiff << std::endl;
    GPU_ASSERT(fwdDiff < 1e-5f, "GAP forward output must match CPU within 1e-5");

    std::vector<float> h_go(N * C, 1.5f);
    Tensor go_cpu({N, C}, h_go, DeviceType::CPU);
    Tensor go_gpu({N, C}, h_go, DeviceType::CUDA);

    Tensor gi_cpu = gap_cpu.backward(go_cpu);
    Tensor gi_gpu = gap_gpu.backward(go_gpu);

    float bwdDiff = maxDiff(gi_cpu.toHostVector(), gi_gpu.toHostVector());
    std::cout << "  - GAP Backward GradInput Max Diff: " << bwdDiff << std::endl;
    GPU_ASSERT(bwdDiff < 1e-5f, "GAP backward gradInput must match CPU within 1e-5");

    std::cout << "[+] GlobalAveragePool2D verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 9: BCEWithLogitsLoss (Forward & Backward)
// -------------------------------------------------------------
bool test_loss_function() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 9: BCEWithLogitsLoss GPU vs CPU (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    BCEWithLogitsLoss crit_cpu, crit_gpu;
    std::vector<float> h_z = {-12.0f, -0.5f, 0.0f, 1.5f, 15.0f};
    std::vector<float> h_y = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    Tensor z_cpu({h_z.size(), 1}, h_z, DeviceType::CPU);
    Tensor z_gpu({h_z.size(), 1}, h_z, DeviceType::CUDA);
    Tensor y_cpu({h_y.size(), 1}, h_y, DeviceType::CPU);
    Tensor y_gpu({h_y.size(), 1}, h_y, DeviceType::CUDA);

    float l_cpu = crit_cpu.forward(z_cpu, y_cpu);
    float l_gpu = crit_gpu.forward(z_gpu, y_gpu);

    float diffLoss = std::abs(l_cpu - l_gpu);
    std::cout << "  - Forward Loss Value Diff: " << diffLoss << " (CPU: " << l_cpu << ", GPU: " << l_gpu << ")" << std::endl;
    GPU_ASSERT(diffLoss < 1e-5f, "BCE loss must match CPU within 1e-5");

    Tensor gz_cpu = crit_cpu.backward();
    Tensor gz_gpu = crit_gpu.backward();

    float diffGrad = maxDiff(gz_cpu.toHostVector(), gz_gpu.toHostVector());
    std::cout << "  - Backward gradLogits Max Diff: " << diffGrad << std::endl;
    GPU_ASSERT(diffGrad < 1e-5f, "BCE gradLogits must match CPU within 1e-5");

    std::cout << "[+] BCEWithLogitsLoss verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 10: SGD Optimizer
// -------------------------------------------------------------
bool test_sgd_optimizer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 10: SGD Parameter Update GPU vs CPU" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::vector<float> h_w = {1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<float> h_g = {0.1f, -0.2f, 0.3f, -0.4f};

    Tensor tw_cpu({4}, h_w, DeviceType::CPU);
    Tensor tw_gpu({4}, h_w, DeviceType::CUDA);

    Parameter p_cpu(tw_cpu, true);
    Parameter p_gpu(tw_gpu, true);

    p_cpu.grad().copyFromHost(h_g.data(), 4);
    p_gpu.grad().copyFromHost(h_g.data(), 4);

    SGD opt_cpu({&p_cpu}, 0.01f, 0.9f, 1e-4f);
    SGD opt_gpu({&p_gpu}, 0.01f, 0.9f, 1e-4f);

    opt_cpu.step();
    opt_gpu.step();

    float diffW = maxDiff(p_cpu.data().toHostVector(), p_gpu.data().toHostVector());
    float diffV = maxDiff(p_cpu.velocity().toHostVector(), p_gpu.velocity().toHostVector());

    std::cout << "  - Post-step Weight Max Diff: " << diffW << std::endl;
    std::cout << "  - Post-step Velocity Max Diff: " << diffV << std::endl;

    GPU_ASSERT(diffW < 1e-5f, "SGD updated weight must match CPU within 1e-5");
    GPU_ASSERT(diffV < 1e-5f, "SGD updated velocity must match CPU within 1e-5");

    std::cout << "[+] SGD Optimizer verified successfully on GPU." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 11: Full CatDogCNN Architecture on Actual GTX 1650
// -------------------------------------------------------------
struct BenchmarkResult {
    size_t batchSize;
    double fwdMs;
    double bwdMs;
    double totalMs;
    float initialLoss;
    float postStepLoss;
    float cpuGpuOutputDiff;
    float cpuGpuGradDiff;
};

bool run_full_cnn_benchmark(size_t batchSize, BenchmarkResult& res) {
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << " Running Full CatDogCNN on NVIDIA GTX 1650 | Batch Size: " << batchSize << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    printMemInfo("Pre-Model");

    // Construct GPU Model
    Conv2D conv1_gpu(3, 32, 3, 1, 1, true, DeviceType::CUDA, "b1.conv");
    ReLU relu1_gpu("b1.relu");
    MaxPool2D pool1_gpu(2, 2, "b1.pool");

    Conv2D conv2_gpu(32, 64, 3, 1, 1, true, DeviceType::CUDA, "b2.conv");
    ReLU relu2_gpu("b2.relu");
    MaxPool2D pool2_gpu(2, 2, "b2.pool");

    Conv2D conv3_gpu(64, 128, 3, 1, 1, true, DeviceType::CUDA, "b3.conv");
    ReLU relu3_gpu("b3.relu");
    MaxPool2D pool3_gpu(2, 2, "b3.pool");

    GlobalAveragePool2D gap_gpu("gap");
    Linear fc1_gpu(128, 32, true, DeviceType::CUDA, "fc1");
    ReLU reluFc_gpu("reluFc");
    Linear fc2_gpu(32, 1, true, DeviceType::CUDA, "fc2");

    BCEWithLogitsLoss criterion_gpu("bce");

    // Construct duplicate CPU Model to verify exact numerical match
    Conv2D conv1_cpu(3, 32, 3, 1, 1, true, DeviceType::CPU, "b1.conv");
    Conv2D conv2_cpu(32, 64, 3, 1, 1, true, DeviceType::CPU, "b2.conv");
    Conv2D conv3_cpu(64, 128, 3, 1, 1, true, DeviceType::CPU, "b3.conv");
    GlobalAveragePool2D gap_cpu("gap");
    ReLU relu1_cpu, relu2_cpu, relu3_cpu, reluFc_cpu;
    MaxPool2D pool1_cpu(2, 2), pool2_cpu(2, 2), pool3_cpu(2, 2);
    Linear fc1_cpu(128, 32, true, DeviceType::CPU, "fc1");
    Linear fc2_cpu(32, 1, true, DeviceType::CPU, "fc2");
    BCEWithLogitsLoss criterion_cpu("bce");

    // Synchronize weights between CPU and GPU
    conv1_cpu.weight().data().copyFrom(conv1_gpu.weight().data());
    conv1_cpu.bias().data().copyFrom(conv1_gpu.bias().data());
    conv2_cpu.weight().data().copyFrom(conv2_gpu.weight().data());
    conv2_cpu.bias().data().copyFrom(conv2_gpu.bias().data());
    conv3_cpu.weight().data().copyFrom(conv3_gpu.weight().data());
    conv3_cpu.bias().data().copyFrom(conv3_gpu.bias().data());
    fc1_cpu.weight().data().copyFrom(fc1_gpu.weight().data());
    fc1_cpu.bias().data().copyFrom(fc1_gpu.bias().data());
    fc2_cpu.weight().data().copyFrom(fc2_gpu.weight().data());
    fc2_cpu.bias().data().copyFrom(fc2_gpu.bias().data());

    std::vector<Parameter*> allParamsGpu;
    for (Parameter* p : conv1_gpu.getParameters()) allParamsGpu.push_back(p);
    for (Parameter* p : conv2_gpu.getParameters()) allParamsGpu.push_back(p);
    for (Parameter* p : conv3_gpu.getParameters()) allParamsGpu.push_back(p);
    for (Parameter* p : fc1_gpu.getParameters()) allParamsGpu.push_back(p);
    for (Parameter* p : fc2_gpu.getParameters()) allParamsGpu.push_back(p);

    SGD optimizer_gpu(allParamsGpu, 0.001f, 0.0f, 0.0f);

    printMemInfo("Post-Model");

    // Synthetic normalized image batch [batchSize, 3, 224, 224]
    std::vector<float> h_in(batchSize * 3 * 224 * 224);
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_in[i] = static_cast<float>((i % 255)) / 255.0f;
    }
    std::vector<float> h_targets(batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        h_targets[i] = static_cast<float>(i % 2); // alternating 0 and 1
    }

    Tensor in_gpu({batchSize, 3, 224, 224}, h_in, DeviceType::CUDA);
    Tensor targets_gpu({batchSize, 1}, h_targets, DeviceType::CUDA);

    Tensor in_cpu({batchSize, 3, 224, 224}, h_in, DeviceType::CPU);
    Tensor targets_cpu({batchSize, 1}, h_targets, DeviceType::CPU);

    printMemInfo("Post-Input-Allocation");

    // 1. Warm-up GPU pass
    {
        Tensor w_out = fc2_gpu.forward(reluFc_gpu.forward(fc1_gpu.forward(
            gap_gpu.forward(pool3_gpu.forward(relu3_gpu.forward(conv3_gpu.forward(
                pool2_gpu.forward(relu2_gpu.forward(conv2_gpu.forward(
                    pool1_gpu.forward(relu1_gpu.forward(conv1_gpu.forward(in_gpu)))))))))))));
        criterion_gpu.forward(w_out, targets_gpu);
        CUDABackend::synchronize();
    }

    // 2. Timed Forward Pass
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor out1 = conv1_gpu.forward(in_gpu);
    Tensor out2 = relu1_gpu.forward(out1);
    Tensor out3 = pool1_gpu.forward(out2);

    Tensor out4 = conv2_gpu.forward(out3);
    Tensor out5 = relu2_gpu.forward(out4);
    Tensor out6 = pool2_gpu.forward(out5);

    Tensor out7 = conv3_gpu.forward(out6);
    Tensor out8 = relu3_gpu.forward(out7);
    Tensor out9 = pool3_gpu.forward(out8);

    Tensor out10 = gap_gpu.forward(out9);
    Tensor out11 = fc1_gpu.forward(out10);
    Tensor out12 = reluFc_gpu.forward(out11);
    Tensor logits_gpu = fc2_gpu.forward(out12);

    float initialLoss = criterion_gpu.forward(logits_gpu, targets_gpu);
    CUDABackend::synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwdMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Verify GPU Forward match against CPU Forward
    Tensor out1_c = conv1_cpu.forward(in_cpu);
    Tensor out2_c = relu1_cpu.forward(out1_c);
    Tensor out3_c = pool1_cpu.forward(out2_c);
    Tensor out4_c = conv2_cpu.forward(out3_c);
    Tensor out5_c = relu2_cpu.forward(out4_c);
    Tensor out6_c = pool2_cpu.forward(out5_c);
    Tensor out7_c = conv3_cpu.forward(out6_c);
    Tensor out8_c = relu3_cpu.forward(out7_c);
    Tensor out9_c = pool3_cpu.forward(out8_c);
    Tensor out10_c = gap_cpu.forward(out9_c);
    Tensor out11_c = fc1_cpu.forward(out10_c);
    Tensor out12_c = reluFc_cpu.forward(out11_c);
    Tensor logits_cpu = fc2_cpu.forward(out12_c);
    criterion_cpu.forward(logits_cpu, targets_cpu);

    float cpuGpuOutDiff = maxDiff(logits_cpu.toHostVector(), logits_gpu.toHostVector());

    // 3. Timed Backward Pass
    optimizer_gpu.zeroGrad();
    auto t2 = std::chrono::high_resolution_clock::now();

    Tensor gLoss = criterion_gpu.backward();
    Tensor gFc2 = fc2_gpu.backward(gLoss);
    Tensor gReluFc = reluFc_gpu.backward(gFc2);
    Tensor gFc1 = fc1_gpu.backward(gReluFc);
    Tensor gGap = gap_gpu.backward(gFc1);

    Tensor gPool3 = pool3_gpu.backward(gGap);
    Tensor gRelu3 = relu3_gpu.backward(gPool3);
    Tensor gConv3 = conv3_gpu.backward(gRelu3);

    Tensor gPool2 = pool2_gpu.backward(gConv3);
    Tensor gRelu2 = relu2_gpu.backward(gPool2);
    Tensor gConv2 = conv2_gpu.backward(gRelu2);

    Tensor gPool1 = pool1_gpu.backward(gConv2);
    Tensor gRelu1 = relu1_gpu.backward(gPool1);
    Tensor gInput = conv1_gpu.backward(gRelu1);

    CUDABackend::synchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwdMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // CPU Backward for comparison
    Tensor gLoss_c = criterion_cpu.backward();
    Tensor gFc2_c = fc2_cpu.backward(gLoss_c);
    Tensor gReluFc_c = reluFc_cpu.backward(gFc2_c);
    Tensor gFc1_c = fc1_cpu.backward(gReluFc_c);
    Tensor gGap_c = gap_cpu.backward(gFc1_c);
    Tensor gPool3_c = pool3_cpu.backward(gGap_c);
    Tensor gRelu3_c = relu3_cpu.backward(gPool3_c);
    Tensor gConv3_c = conv3_cpu.backward(gRelu3_c);
    Tensor gPool2_c = pool2_cpu.backward(gConv3_c);
    Tensor gRelu2_c = relu2_cpu.backward(gPool2_c);
    Tensor gConv2_c = conv2_cpu.backward(gRelu2_c);
    Tensor gPool1_c = pool1_cpu.backward(gConv2_c);
    Tensor gRelu1_c = relu1_cpu.backward(gPool1_c);
    Tensor gInput_c = conv1_cpu.backward(gRelu1_c);

    float cpuGpuGradDiff = maxDiff(gInput_c.toHostVector(), gInput.toHostVector());

    // 4. SGD Parameter Update
    auto t4 = std::chrono::high_resolution_clock::now();
    optimizer_gpu.step();
    CUDABackend::synchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double stepMs = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // Re-run forward pass to verify loss reduction
    Tensor postLogits = fc2_gpu.forward(reluFc_gpu.forward(fc1_gpu.forward(
        gap_gpu.forward(pool3_gpu.forward(relu3_gpu.forward(conv3_gpu.forward(
            pool2_gpu.forward(relu2_gpu.forward(conv2_gpu.forward(
                pool1_gpu.forward(relu1_gpu.forward(conv1_gpu.forward(in_gpu)))))))))))));
    float postStepLoss = criterion_gpu.forward(postLogits, targets_gpu);

    double totalMs = fwdMs + bwdMs + stepMs;

    res.batchSize = batchSize;
    res.fwdMs = fwdMs;
    res.bwdMs = bwdMs;
    res.totalMs = totalMs;
    res.initialLoss = initialLoss;
    res.postStepLoss = postStepLoss;
    res.cpuGpuOutputDiff = cpuGpuOutDiff;
    res.cpuGpuGradDiff = cpuGpuGradDiff;

    std::cout << "  - Initial BCE Loss: " << initialLoss << std::endl;
    std::cout << "  - Post-Step BCE Loss: " << postStepLoss << " (change: " << (postStepLoss - initialLoss) << ")" << std::endl;
    std::cout << "  - Forward Pass Time: " << fwdMs << " ms" << std::endl;
    std::cout << "  - Backward Pass Time: " << bwdMs << " ms" << std::endl;
    std::cout << "  - Full Training Step Time: " << totalMs << " ms" << std::endl;
    std::cout << "  - CPU vs GPU Logits Max Diff: " << cpuGpuOutDiff << std::endl;
    std::cout << "  - CPU vs GPU Input Gradient Max Diff: " << cpuGpuGradDiff << std::endl;

    GPU_ASSERT(postStepLoss < initialLoss, "Optimization step must reduce loss on GPU");
    GPU_ASSERT(cpuGpuOutDiff < 5e-4f, "GPU forward pass must match CPU reference");
    GPU_ASSERT(cpuGpuGradDiff < 5e-4f, "GPU backward pass must match CPU reference");

    printMemInfo("Final-After-Run");
    return true;
}

int main() {
    try {
        std::cout << "==========================================================" << std::endl;
        std::cout << "   CatDogCNN: Phase 3 Actual GPU Execution Verification   " << std::endl;
        std::cout << "==========================================================" << std::endl;

        bool ok = true;
        if (!test_device_query()) ok = false;
        if (!test_memory_transfers()) ok = false;
        if (!test_cublas_sgemm()) ok = false;
        if (!test_linear_layer()) ok = false;
        if (!test_conv2d_layer()) ok = false;
        if (!test_relu_layer()) ok = false;
        if (!test_maxpool_layer()) ok = false;
        if (!test_gap_layer()) ok = false;
        if (!test_loss_function()) ok = false;
        if (!test_sgd_optimizer()) ok = false;

        if (!ok) {
            std::cerr << "[-] Unit operation tests failed on GPU!" << std::endl;
            return 1;
        }

        std::vector<BenchmarkResult> benchmarks;
        std::vector<size_t> batchSizes = {2, 4, 8};

        for (size_t b : batchSizes) {
            BenchmarkResult res;
            if (!run_full_cnn_benchmark(b, res)) {
                std::cerr << "[-] Benchmark failed for batch size " << b << std::endl;
                return 1;
            }
            benchmarks.push_back(res);
        }

        std::cout << "\n==========================================================" << std::endl;
        std::cout << "                  BENCHMARK SUMMARY TABLE                 " << std::endl;
        std::cout << "==========================================================" << std::endl;
        std::cout << "| Batch | Fwd (ms) | Bwd (ms) | Total (ms) | Initial Loss | Post Loss  | Loss Delta  |" << std::endl;
        std::cout << "|-------|----------|----------|------------|--------------|------------|-------------|" << std::endl;
        for (const auto& b : benchmarks) {
            std::cout << "|   " << b.batchSize << "   | "
                      << std::setw(8) << std::fixed << std::setprecision(2) << b.fwdMs << " | "
                      << std::setw(8) << std::fixed << std::setprecision(2) << b.bwdMs << " | "
                      << std::setw(10) << std::fixed << std::setprecision(2) << b.totalMs << " | "
                      << std::setw(12) << std::fixed << std::setprecision(6) << b.initialLoss << " | "
                      << std::setw(10) << std::fixed << std::setprecision(6) << b.postStepLoss << " | "
                      << std::setw(11) << std::fixed << std::setprecision(6) << (b.postStepLoss - b.initialLoss) << " |" << std::endl;
        }
        std::cout << "==========================================================" << std::endl;
        std::cout << "  ALL GPU VERIFICATION TESTS & BENCHMARKS PASSED (100%)   " << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[-] CRITICAL EXCEPTION CAUGHT: " << e.what() << std::endl;
        return 1;
    }
}
