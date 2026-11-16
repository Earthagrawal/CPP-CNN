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

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

static bool approxEqual(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) <= tol * (1.0f + std::max(std::abs(a), std::abs(b)));
}

// -------------------------------------------------------------
// Test 1: Environment & CUDA Availability
// -------------------------------------------------------------
bool test_cuda_environment() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 1: CUDA Environment & Hardware Check" << std::endl;
    std::cout << "==========================================" << std::endl;

    bool available = CUDABackend::isAvailable();
    int count = CUDABackend::getDeviceCount();
    std::cout << "[*] CUDA Available via Runtime API: " << (available ? "YES" : "NO") << std::endl;
    std::cout << "[*] CUDA Device Count: " << count << std::endl;

    if (available && count > 0) {
        for (int i = 0; i < count; ++i) {
            CUDADeviceInfo info = CUDABackend::getDeviceInfo(i);
            std::cout << "  - Device " << i << ": " << info.name
                      << " | VRAM: " << (info.totalGlobalMem / (1024 * 1024)) << " MB"
                      << " | Compute Capability: " << info.major << "." << info.minor
                      << " | SMs: " << info.multiProcessorCount << std::endl;
        }
    } else {
        std::cout << "[i] System running on CPU fallback mode (CUDA runtime / toolkit not active in current build)." << std::endl;
    }
    return true;
}

// -------------------------------------------------------------
// Test 2: Tensor & Matrix GEMM Operations
// -------------------------------------------------------------
bool test_tensor_and_matrix() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 2: Tensor & Matrix GEMM Operations" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. Tensor shape & indexing
    Tensor t({2, 3, 4}, 1.5f);
    TEST_ASSERT(t.size() == 24, "Tensor total size calculation");
    TEST_ASSERT(t.strides()[0] == 12 && t.strides()[1] == 4 && t.strides()[2] == 1, "Tensor strides calculation");
    TEST_ASSERT(approxEqual(t.at({1, 2, 3}), 1.5f), "Tensor multidimensional indexing");

    // 2. Matrix Matmul verification: C = A * B
    // A: 2x3, B: 3x2
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 1], [2, 3]]
    // C = [[1*7+2*9+3*2, 1*8+2*1+3*3], [4*7+5*9+6*2, 4*8+5*1+6*3]]
    //   = [[7+18+6, 8+2+9], [28+45+12, 32+5+18]]
    //   = [[31, 19], [85, 55]]
    Matrix A(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix B(3, 2, {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f});
    Matrix C = Matrix::matmul(A, B);

    TEST_ASSERT(C.rows() == 2 && C.cols() == 2, "Matrix matmul output shape");
    TEST_ASSERT(approxEqual(C(0, 0), 31.0f), "C(0,0) == 31.0");
    TEST_ASSERT(approxEqual(C(0, 1), 19.0f), "C(0,1) == 19.0");
    TEST_ASSERT(approxEqual(C(1, 0), 85.0f), "C(1,0) == 85.0");
    TEST_ASSERT(approxEqual(C(1, 1), 55.0f), "C(1,1) == 55.0");

    std::cout << "[+] Tensor & Matrix GEMM verified successfully." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 3: Layer Unit Tests (Linear, ReLU, MaxPool2D, GAP, Conv2D)
// -------------------------------------------------------------
bool test_layers_unit() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 3: Layer Unit Tests (Fwd & Bwd)" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. ReLU
    {
        ReLU relu;
        Tensor in({1, 4}, {-2.0f, 0.0f, 3.5f, -0.1f});
        Tensor out = relu.forward(in);
        TEST_ASSERT(approxEqual(out[0], 0.0f) && approxEqual(out[1], 0.0f) &&
                    approxEqual(out[2], 3.5f) && approxEqual(out[3], 0.0f), "ReLU forward");

        Tensor gradOut({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
        Tensor gradIn = relu.backward(gradOut);
        TEST_ASSERT(approxEqual(gradIn[0], 0.0f) && approxEqual(gradIn[1], 0.0f) &&
                    approxEqual(gradIn[2], 3.0f) && approxEqual(gradIn[3], 0.0f), "ReLU backward");
        std::cout << "[+] ReLU forward and backward passed." << std::endl;
    }

    // 2. Linear Layer
    {
        Linear fc(3, 2, true, DeviceType::CPU, "TestLinear");
        // Manually set weight: [[1, 2, 3], [4, 5, 6]]
        fc.weight().data().copyFromHost(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}.data(), 6);
        // Manually set bias: [0.5, -0.5]
        fc.bias().data().copyFromHost(std::vector<float>{0.5f, -0.5f}.data(), 2);

        // Input: [[1, 0, 2]] (batch size 1)
        // Y = [1*1 + 0*2 + 2*3 + 0.5, 1*4 + 0*5 + 2*6 - 0.5] = [7.5, 15.5]
        Tensor in({1, 3}, {1.0f, 0.0f, 2.0f});
        Tensor out = fc.forward(in);
        TEST_ASSERT(approxEqual(out[0], 7.5f) && approxEqual(out[1], 15.5f), "Linear forward output");

        // Backward: gradOut = [[1.0, 1.0]]
        // gradIn = gradOut * W = [1*1 + 1*4, 1*2 + 1*5, 1*3 + 1*6] = [5, 7, 9]
        fc.weight().zeroGrad();
        fc.bias().zeroGrad();
        Tensor gradIn = fc.backward(Tensor({1, 2}, {1.0f, 1.0f}));
        TEST_ASSERT(approxEqual(gradIn[0], 5.0f) && approxEqual(gradIn[1], 7.0f) && approxEqual(gradIn[2], 9.0f), "Linear backward gradInput");
        TEST_ASSERT(approxEqual(fc.bias().grad()[0], 1.0f) && approxEqual(fc.bias().grad()[1], 1.0f), "Linear backward gradBias");
        std::cout << "[+] Linear layer forward and backward passed." << std::endl;
    }

    // 3. MaxPool2D
    {
        MaxPool2D pool(2, 2);
        // Input: 1 batch, 1 channel, 4x4 matrix
        // [ 1,  3,  2,  4 ]
        // [ 5,  6,  7,  8 ]
        // [ 9, 10, 11, 12 ]
        // [13, 14, 15, 16 ]
        std::vector<float> inData = {
            1,  3,  2,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        Tensor in({1, 1, 4, 4}, inData);
        Tensor out = pool.forward(in);
        // Out should be 2x2:
        // top-left max: 6, top-right max: 8
        // bottom-left max: 14, bottom-right max: 16
        TEST_ASSERT(out.shape()[2] == 2 && out.shape()[3] == 2, "MaxPool2D output shape");
        TEST_ASSERT(approxEqual(out[0], 6.0f) && approxEqual(out[1], 8.0f) &&
                    approxEqual(out[2], 14.0f) && approxEqual(out[3], 16.0f), "MaxPool2D forward values");

        Tensor gradOut({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
        Tensor gradIn = pool.backward(gradOut);
        TEST_ASSERT(approxEqual(gradIn[5], 1.0f), "MaxPool2D backward argmax routing 1");
        TEST_ASSERT(approxEqual(gradIn[7], 2.0f), "MaxPool2D backward argmax routing 2");
        TEST_ASSERT(approxEqual(gradIn[13], 3.0f), "MaxPool2D backward argmax routing 3");
        TEST_ASSERT(approxEqual(gradIn[15], 4.0f), "MaxPool2D backward argmax routing 4");
        std::cout << "[+] MaxPool2D forward and backward passed." << std::endl;
    }

    // 4. GlobalAveragePool2D
    {
        GlobalAveragePool2D gap;
        // Input: 1 batch, 2 channels, 2x2 spatial
        // ch0: [1, 2, 3, 4] -> avg = 2.5
        // ch1: [10, 20, 30, 40] -> avg = 25.0
        Tensor in({1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f});
        Tensor out = gap.forward(in);
        TEST_ASSERT(out.shape()[0] == 1 && out.shape()[1] == 2, "GAP output shape");
        TEST_ASSERT(approxEqual(out[0], 2.5f) && approxEqual(out[1], 25.0f), "GAP forward values");

        Tensor gradIn = gap.backward(Tensor({1, 2}, {4.0f, 8.0f}));
        // Each element of ch0 should receive 4.0 / 4 = 1.0
        // Each element of ch1 should receive 8.0 / 4 = 2.0
        for (size_t i = 0; i < 4; ++i) {
            TEST_ASSERT(approxEqual(gradIn[i], 1.0f), "GAP backward ch0 gradient");
            TEST_ASSERT(approxEqual(gradIn[4 + i], 2.0f), "GAP backward ch1 gradient");
        }
        std::cout << "[+] GlobalAveragePool2D forward and backward passed." << std::endl;
    }

    // 5. Conv2D
    {
        // 1 inChannel, 1 outChannel, 3x3 kernel, stride 1, padding 1
        Conv2D conv(1, 1, 3, 1, 1, false, DeviceType::CPU, "TestConv");
        // All-ones 3x3 filter
        conv.weight().data().fill(1.0f);
        // All-ones 3x3 input
        Tensor in({1, 1, 3, 3}, 1.0f);
        Tensor out = conv.forward(in);
        // Corners have 4 neighbors (padded), edges have 6, center has 9
        // out[0, 0, 0, 0] = corner = 4.0
        // out[0, 0, 0, 1] = edge = 6.0
        // out[0, 0, 1, 1] = center = 9.0
        TEST_ASSERT(approxEqual(out.at({0, 0, 0, 0}), 4.0f), "Conv2D corner padded value");
        TEST_ASSERT(approxEqual(out.at({0, 0, 0, 1}), 6.0f), "Conv2D edge padded value");
        TEST_ASSERT(approxEqual(out.at({0, 0, 1, 1}), 9.0f), "Conv2D center padded value");

        Tensor gradIn = conv.backward(out);
        TEST_ASSERT(gradIn.shape() == in.shape(), "Conv2D gradInput shape matches input");
        std::cout << "[+] Conv2D forward and backward passed." << std::endl;
    }

    return true;
}

// -------------------------------------------------------------
// Test 4: Loss & Optimizer Numerical Verification
// -------------------------------------------------------------
bool test_loss_and_optimizer() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 4: Loss & Optimizer Numerical Verification" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. BCEWithLogitsLoss
    {
        BCEWithLogitsLoss criterion;
        // Logits: [0.0], Target: [1.0] -> Sigmoid(0) = 0.5 -> BCE = -log(0.5) = 0.693147
        Tensor logits({1, 1}, 0.0f);
        Tensor targets({1, 1}, 1.0f);
        float loss = criterion.forward(logits, targets);
        TEST_ASSERT(approxEqual(loss, 0.693147f, 1e-3f), "BCEWithLogitsLoss forward at 0.0");

        // Backward: grad = (sigma(z) - y) / N = (0.5 - 1.0) / 1 = -0.5
        Tensor gradLogits = criterion.backward();
        TEST_ASSERT(approxEqual(gradLogits[0], -0.5f), "BCEWithLogitsLoss backward at 0.0");

        // Test extreme logits for numerical stability (no NaN/Inf)
        Tensor extremeLogits({2, 1}, {50.0f, -50.0f});
        Tensor extremeTargets({2, 1}, {1.0f, 0.0f});
        float extremeLoss = criterion.forward(extremeLogits, extremeTargets);
        TEST_ASSERT(!std::isnan(extremeLoss) && !std::isinf(extremeLoss) && extremeLoss < 1e-5f, "BCE numerical stability under extreme logits");
        std::cout << "[+] BCEWithLogitsLoss forward, backward & extreme stability passed." << std::endl;
    }

    // 2. SGD Optimizer
    {
        Tensor paramTensor({2}, {10.0f, -5.0f});
        Parameter p(paramTensor, true, "w");
        p.grad().copyFromHost(std::vector<float>{1.0f, -2.0f}.data(), 2);

        SGD optimizer({&p}, 0.1f, 0.0f, 0.0f); // lr=0.1, no momentum
        optimizer.step();

        // New w: 10 - 0.1*1.0 = 9.9, -5 - 0.1*(-2.0) = -4.8
        TEST_ASSERT(approxEqual(p.data()[0], 9.9f), "SGD step index 0");
        TEST_ASSERT(approxEqual(p.data()[1], -4.8f), "SGD step index 1");

        optimizer.zeroGrad();
        TEST_ASSERT(p.grad()[0] == 0.0f && p.grad()[1] == 0.0f, "SGD zeroGrad");
        std::cout << "[+] SGD step and zeroGrad passed." << std::endl;
    }

    return true;
}

// -------------------------------------------------------------
// Test 5: Full CNN Pipeline Forward & Backward Verification
// Architecture:
// Input (2, 3, 224, 224)
// Block 1: Conv2D(3->32, 3x3, s1, p1) -> ReLU -> MaxPool2D(2x2, s2) -> (2, 32, 112, 112)
// Block 2: Conv2D(32->64, 3x3, s1, p1) -> ReLU -> MaxPool2D(2x2, s2) -> (2, 64, 56, 56)
// Block 3: Conv2D(64->128, 3x3, s1, p1) -> ReLU -> MaxPool2D(2x2, s2) -> (2, 128, 28, 28)
// GlobalAveragePool2D -> (2, 128)
// Linear Head: Linear(128->32) -> ReLU -> Linear(32->1) -> (2, 1)
// Loss: BCEWithLogitsLoss
// -------------------------------------------------------------
bool test_full_cnn_architecture() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 5: Complete CatDogCNN Architecture Verification" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::cout << "[*] Constructing CatDogCNN modular layers..." << std::endl;
    // Block 1
    Conv2D conv1(3, 32, 3, 1, 1, true, DeviceType::CPU, "b1.conv");
    ReLU relu1("b1.relu");
    MaxPool2D pool1(2, 2, "b1.pool");

    // Block 2
    Conv2D conv2(32, 64, 3, 1, 1, true, DeviceType::CPU, "b2.conv");
    ReLU relu2("b2.relu");
    MaxPool2D pool2(2, 2, "b2.pool");

    // Block 3
    Conv2D conv3(64, 128, 3, 1, 1, true, DeviceType::CPU, "b3.conv");
    ReLU relu3("b3.relu");
    MaxPool2D pool3(2, 2, "b3.pool");

    // Head
    GlobalAveragePool2D gap("gap");
    Linear fc1(128, 32, true, DeviceType::CPU, "fc1");
    ReLU reluFc("reluFc");
    Linear fc2(32, 1, true, DeviceType::CPU, "fc2");

    BCEWithLogitsLoss criterion("bce");

    // Collect all parameters
    std::vector<Parameter*> allParams;
    for (Parameter* p : conv1.getParameters()) allParams.push_back(p);
    for (Parameter* p : conv2.getParameters()) allParams.push_back(p);
    for (Parameter* p : conv3.getParameters()) allParams.push_back(p);
    for (Parameter* p : fc1.getParameters()) allParams.push_back(p);
    for (Parameter* p : fc2.getParameters()) allParams.push_back(p);

    size_t totalParameters = 0;
    for (Parameter* p : allParams) {
        totalParameters += p->data().size();
    }
    std::cout << "[*] Total trainable parameters in CatDogCNN: " << totalParameters << std::endl;

    SGD optimizer(allParams, 0.001f, 0.0f, 0.0f);

    // Test with Batch size = 2, RGB images [2, 3, 224, 224]
    // Normalized to [0, 1]
    std::cout << "[*] Initializing synthetic input batch: [2, 3, 224, 224]..." << std::endl;
    size_t batchSize = 2;
    Tensor input({batchSize, 3, 224, 224}, 0.5f);
    // Add small variation to inputs
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 255)) / 255.0f;
    }

    // Binary targets: [Cat = 0.0, Dog = 1.0]
    Tensor targets({batchSize, 1}, {0.0f, 1.0f});

    // ---------------- Forward Pass ----------------
    std::cout << "[*] Executing forward pass..." << std::endl;
    auto tStart = std::chrono::high_resolution_clock::now();

    Tensor b1_c = conv1.forward(input);
    TEST_ASSERT(b1_c.shape() == std::vector<size_t>({batchSize, 32, 224, 224}), "Conv1 output shape");
    Tensor b1_r = relu1.forward(b1_c);
    Tensor b1_p = pool1.forward(b1_r);
    TEST_ASSERT(b1_p.shape() == std::vector<size_t>({batchSize, 32, 112, 112}), "Block 1 output shape [2, 32, 112, 112]");

    Tensor b2_c = conv2.forward(b1_p);
    TEST_ASSERT(b2_c.shape() == std::vector<size_t>({batchSize, 64, 112, 112}), "Conv2 output shape");
    Tensor b2_r = relu2.forward(b2_c);
    Tensor b2_p = pool2.forward(b2_r);
    TEST_ASSERT(b2_p.shape() == std::vector<size_t>({batchSize, 64, 56, 56}), "Block 2 output shape [2, 64, 56, 56]");

    Tensor b3_c = conv3.forward(b2_p);
    TEST_ASSERT(b3_c.shape() == std::vector<size_t>({batchSize, 128, 56, 56}), "Conv3 output shape");
    Tensor b3_r = relu3.forward(b3_c);
    Tensor b3_p = pool3.forward(b3_r);
    TEST_ASSERT(b3_p.shape() == std::vector<size_t>({batchSize, 128, 28, 28}), "Block 3 output shape [2, 128, 28, 28]");

    Tensor gap_out = gap.forward(b3_p);
    TEST_ASSERT(gap_out.shape() == std::vector<size_t>({batchSize, 128}), "GAP output shape [2, 128]");

    Tensor fc1_out = fc1.forward(gap_out);
    TEST_ASSERT(fc1_out.shape() == std::vector<size_t>({batchSize, 32}), "FC1 output shape [2, 32]");
    Tensor fc1_r = reluFc.forward(fc1_out);

    Tensor logits = fc2.forward(fc1_r);
    TEST_ASSERT(logits.shape() == std::vector<size_t>({batchSize, 1}), "FC2 logits output shape [2, 1]");

    float initialLoss = criterion.forward(logits, targets);
    auto tEnd = std::chrono::high_resolution_clock::now();
    double fwdMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    std::cout << "  - Forward Pass Time: " << fwdMs << " ms" << std::endl;
    std::cout << "  - Initial Logits: [" << logits[0] << ", " << logits[1] << "]" << std::endl;
    std::cout << "  - Initial BCE Loss: " << initialLoss << std::endl;
    TEST_ASSERT(!std::isnan(initialLoss) && !std::isinf(initialLoss), "Initial loss is valid finite float");

    // ---------------- Backward Pass ----------------
    std::cout << "[*] Executing backward pass across entire network..." << std::endl;
    optimizer.zeroGrad();
    auto tbStart = std::chrono::high_resolution_clock::now();

    Tensor gradLoss = criterion.backward();
    Tensor gradFc2 = fc2.backward(gradLoss);
    Tensor gradReluFc = reluFc.backward(gradFc2);
    Tensor gradFc1 = fc1.backward(gradReluFc);
    Tensor gradGap = gap.backward(gradFc1);

    Tensor gradB3_p = pool3.backward(gradGap);
    Tensor gradB3_r = relu3.backward(gradB3_p);
    Tensor gradB3_c = conv3.backward(gradB3_r);

    Tensor gradB2_p = pool2.backward(gradB3_c);
    Tensor gradB2_r = relu2.backward(gradB2_p);
    Tensor gradB2_c = conv2.backward(gradB2_r);

    Tensor gradB1_p = pool1.backward(gradB2_c);
    Tensor gradB1_r = relu1.backward(gradB1_p);
    Tensor gradInput = conv1.backward(gradB1_r);

    auto tbEnd = std::chrono::high_resolution_clock::now();
    double bwdMs = std::chrono::duration<double, std::milli>(tbEnd - tbStart).count();

    std::cout << "  - Backward Pass Time: " << bwdMs << " ms" << std::endl;
    TEST_ASSERT(gradInput.shape() == input.shape(), "gradInput shape matches input");

    // Verify all parameter gradients are computed and non-zero
    for (Parameter* p : allParams) {
        float gradNorm = 0.0f;
        std::vector<float> gVals = p->grad().toHostVector();
        for (float v : gVals) {
            TEST_ASSERT(!std::isnan(v) && !std::isinf(v), "Gradient values are finite");
            gradNorm += v * v;
        }
        std::cout << "  - Parameter '" << p->name() << "' (" << p->data().size()
                  << " elements) Grad L2-Norm: " << std::sqrt(gradNorm) << std::endl;
        TEST_ASSERT(std::sqrt(gradNorm) > 0.0f, "Parameter gradient is non-zero");
    }

    // ---------------- Optimizer Step & Loss Reduction ----------------
    std::cout << "[*] Taking SGD optimizer step..." << std::endl;
    optimizer.step();

    // Re-run forward pass to verify optimization step reduces loss
    Tensor next_out = fc2.forward(reluFc.forward(fc1.forward(
        gap.forward(pool3.forward(relu3.forward(conv3.forward(
            pool2.forward(relu2.forward(conv2.forward(
                pool1.forward(relu1.forward(conv1.forward(input)))))))))))));
    float stepLoss = criterion.forward(next_out, targets);

    std::cout << "  - Post-step Logits: [" << next_out[0] << ", " << next_out[1] << "]" << std::endl;
    std::cout << "  - Post-step BCE Loss: " << stepLoss << " (change: " << (stepLoss - initialLoss) << ")" << std::endl;
    TEST_ASSERT(stepLoss < initialLoss, "Optimization step strictly reduces BCE loss");

    std::cout << "[+] Full CNN forward, backward, and optimization pipeline verified successfully!" << std::endl;
    return true;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "      CatDogCNN: CNN Foundation & CUDA Backend Test       " << std::endl;
    std::cout << "==========================================================" << std::endl;

    bool allPassed = true;

    if (!test_cuda_environment()) allPassed = false;
    if (!test_tensor_and_matrix()) allPassed = false;
    if (!test_layers_unit()) allPassed = false;
    if (!test_loss_and_optimizer()) allPassed = false;
    if (!test_full_cnn_architecture()) allPassed = false;

    std::cout << "\n==========================================================" << std::endl;
    if (allPassed) {
        std::cout << "   ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! (PASS)  " << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    } else {
        std::cerr << "   SOME TESTS FAILED! (FAIL)                              " << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 1;
    }
}
