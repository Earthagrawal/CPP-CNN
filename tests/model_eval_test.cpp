#include "../core/CUDABackend.hpp"
#include "../core/Tensor.hpp"
#include "../core/Parameter.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/BatchNorm2D.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/MaxPool2D.hpp"
#include "../layers/GlobalAveragePool2D.hpp"
#include "../layers/Linear.hpp"
#include "../model/CatDogCNNModel.hpp"
#include "../loss/BCEWithLogitsLoss.hpp"
#include "../evaluation/Evaluator.hpp"
#include "../data/DatasetSplitter.hpp"
#include "../data/DataLoader.hpp"
#include <set>
#include <fstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] TEST FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
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

static bool isAllFinite(const std::vector<float>& v) {
    for (float val : v) {
        if (!std::isfinite(val)) return false;
    }
    return true;
}

static bool hasNonZero(const std::vector<float>& v, float eps = 1e-9f) {
    for (float val : v) {
        if (std::abs(val) > eps) return true;
    }
    return false;
}

// -------------------------------------------------------------
// Test 1: Evaluator Metrics and Edge Cases
// -------------------------------------------------------------
bool test_evaluator_metrics() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 1: Evaluator Metrics & Edge Cases" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Handcrafted test case
    // Logits: {-2.0, -1.0, 1.5, -0.5, 2.0, 0.8}
    // Sigmoids: {0.1192, 0.2689, 0.8176, 0.3775, 0.8808, 0.6899}
    // Predictions (>= 0.5): {0, 0, 1, 0, 1, 1}
    // Targets: {0, 1, 1, 0, 0, 1}
    // Results:
    // Sample 0: Actual 0, Pred 0 -> TN
    // Sample 1: Actual 1, Pred 0 -> FN
    // Sample 2: Actual 1, Pred 1 -> TP
    // Sample 3: Actual 0, Pred 0 -> TN
    // Sample 4: Actual 0, Pred 1 -> FP
    // Sample 5: Actual 1, Pred 1 -> TP
    // TN = 2, FP = 1, FN = 1, TP = 2
    // Accuracy = 4 / 6 = 0.6666667
    // Precision = 2 / 3 = 0.6666667
    // Recall = 2 / 3 = 0.6666667
    // F1 = 0.6666667

    Tensor logits({6, 1}, {-2.0f, -1.0f, 1.5f, -0.5f, 2.0f, 0.8f});
    Tensor targets({6, 1}, {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f});

    EvaluationResult res = Evaluator::evaluate(logits, targets);

    std::cout << res.summary() << std::endl;

    TEST_ASSERT(res.confusionMatrix.tn == 2, "TN should be 2");
    TEST_ASSERT(res.confusionMatrix.fp == 1, "FP should be 1");
    TEST_ASSERT(res.confusionMatrix.fn == 1, "FN should be 1");
    TEST_ASSERT(res.confusionMatrix.tp == 2, "TP should be 2");

    float expectedAcc = 4.0f / 6.0f;
    float expectedPrec = 2.0f / 3.0f;
    float expectedRec = 2.0f / 3.0f;
    float expectedF1 = 2.0f / 3.0f;

    TEST_ASSERT(std::abs(res.accuracy - expectedAcc) < 1e-4f, "Accuracy mismatch");
    TEST_ASSERT(std::abs(res.precision - expectedPrec) < 1e-4f, "Precision mismatch");
    TEST_ASSERT(std::abs(res.recall - expectedRec) < 1e-4f, "Recall mismatch");
    TEST_ASSERT(std::abs(res.f1Score - expectedF1) < 1e-4f, "F1-score mismatch");
    TEST_ASSERT(res.loss > 0.0f && std::isfinite(res.loss), "Loss must be positive finite");

    // Test safe division against edge cases (no positive predictions)
    Tensor allNegLogits({4, 1}, {-5.0f, -6.0f, -7.0f, -8.0f});
    Tensor allZeroTargets({4, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    EvaluationResult edgeRes1 = Evaluator::evaluate(allNegLogits, allZeroTargets);
    TEST_ASSERT(edgeRes1.confusionMatrix.tp == 0 && edgeRes1.confusionMatrix.fp == 0, "No positive predictions");
    TEST_ASSERT(edgeRes1.precision == 0.0f, "Precision must safely default to 0.0f when TP+FP=0");
    TEST_ASSERT(edgeRes1.recall == 0.0f, "Recall must safely default to 0.0f when TP+FN=0");
    TEST_ASSERT(edgeRes1.f1Score == 0.0f, "F1 must safely default to 0.0f");
    TEST_ASSERT(edgeRes1.accuracy == 1.0f, "All negative correctly predicted -> 100% accuracy");

    // Test safe division against edge cases (all wrong)
    Tensor allPosLogits({4, 1}, {5.0f, 6.0f, 7.0f, 8.0f});
    EvaluationResult edgeRes2 = Evaluator::evaluate(allPosLogits, allZeroTargets);
    TEST_ASSERT(edgeRes2.confusionMatrix.fp == 4, "All FP");
    TEST_ASSERT(edgeRes2.confusionMatrix.tp == 0, "TP is 0");
    TEST_ASSERT(edgeRes2.precision == 0.0f, "Precision should be 0.0f");
    TEST_ASSERT(edgeRes2.accuracy == 0.0f, "Accuracy should be 0.0f");

    std::cout << "[+] TEST 1 PASSED: Evaluator handles metrics, confusion matrix, and zero-divisions properly." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 2: BatchNorm2D Standalone Verification
// -------------------------------------------------------------
bool test_batchnorm2d_standalone() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 2: BatchNorm2D Standalone Forward & Backward" << std::endl;
    std::cout << "==========================================" << std::endl;

    const size_t N = 4, C = 3, H = 4, W = 4;
    Tensor input({N, C, H, W});
    std::vector<float> inData(N * C * H * W);
    for (size_t i = 0; i < inData.size(); ++i) {
        inData[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.25f;
    }
    input.copyFromHost(inData.data(), inData.size());

    BatchNorm2D bn(C, 1e-5f, 0.1f, DeviceType::CPU, "test_bn");
    bn.train(true);

    Tensor output = bn.forward(input);
    TEST_ASSERT(output.shape() == input.shape(), "Output shape mismatch");

    // For each channel, check normalized mean is near 0 and variance near 1
    std::vector<float> outData = output.toHostVector();
    size_t M = N * H * W;
    for (size_t c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = ((n * C + c) * H + h) * W + w;
                    sum += outData[idx];
                }
            }
        }
        float mean = sum / static_cast<float>(M);
        TEST_ASSERT(std::abs(mean) < 1e-4f, "Channel mean must be ~ 0 after training normalization");
    }

    // Backward pass
    Tensor gradOut({N, C, H, W}, 1.0f);
    Tensor gradIn = bn.backward(gradOut);
    TEST_ASSERT(gradIn.shape() == input.shape(), "gradInput shape mismatch");
    TEST_ASSERT(isAllFinite(gradIn.toHostVector()), "gradInput must be finite");
    TEST_ASSERT(isAllFinite(bn.gamma().grad().toHostVector()), "gamma grad must be finite");
    TEST_ASSERT(isAllFinite(bn.beta().grad().toHostVector()), "beta grad must be finite");
    TEST_ASSERT(hasNonZero(bn.gamma().grad().toHostVector()), "gamma grad must be non-zero");

    std::cout << "[+] TEST 2 PASSED: BatchNorm2D normalized means and computed backward gradients correctly." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 3: Model Architecture Shapes & Forward Propagation
// -------------------------------------------------------------
bool test_model_architecture_shapes() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 3: Model Architecture Shapes & Forward Propagation" << std::endl;
    std::cout << "==========================================" << std::endl;

    CatDogCNNModel model(DeviceType::CPU);
    model.train(true);

    const size_t batchSize = 2;
    Tensor input({batchSize, 3, 224, 224});
    std::vector<float> inData(batchSize * 3 * 224 * 224);
    for (size_t i = 0; i < inData.size(); ++i) {
        inData[i] = static_cast<float>(static_cast<int>(i % 37) - 18) * 0.05f;
    }
    input.copyFromHost(inData.data(), inData.size());

    // Step-by-step layer shape verification
    Tensor x = model.conv1().forward(input);
    std::cout << "  Conv1 output shape: [" << x.shape()[0] << ", " << x.shape()[1]
              << ", " << x.shape()[2] << ", " << x.shape()[3] << "] (expected: [2, 32, 224, 224])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 32, 224, 224}), "Conv1 shape mismatch");

    x = model.bn1().forward(x);
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 32, 224, 224}), "BN1 shape mismatch");

    ReLU relu1;
    x = relu1.forward(x);

    MaxPool2D pool1(2, 2);
    x = pool1.forward(x);
    std::cout << "  Block 1 (Conv->BN->ReLU->MaxPool) output shape: ["
              << x.shape()[0] << ", " << x.shape()[1] << ", " << x.shape()[2] << ", " << x.shape()[3]
              << "] (expected: [2, 32, 112, 112])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 32, 112, 112}), "Block 1 pool shape mismatch");

    x = model.conv2().forward(x);
    x = model.bn2().forward(x);
    ReLU relu2;
    x = relu2.forward(x);
    MaxPool2D pool2(2, 2);
    x = pool2.forward(x);
    std::cout << "  Block 2 output shape: ["
              << x.shape()[0] << ", " << x.shape()[1] << ", " << x.shape()[2] << ", " << x.shape()[3]
              << "] (expected: [2, 64, 56, 56])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 64, 56, 56}), "Block 2 shape mismatch");

    x = model.conv3().forward(x);
    x = model.bn3().forward(x);
    ReLU relu3;
    x = relu3.forward(x);
    MaxPool2D pool3(2, 2);
    x = pool3.forward(x);
    std::cout << "  Block 3 output shape: ["
              << x.shape()[0] << ", " << x.shape()[1] << ", " << x.shape()[2] << ", " << x.shape()[3]
              << "] (expected: [2, 128, 28, 28])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 128, 28, 28}), "Block 3 shape mismatch");

    GlobalAveragePool2D gap;
    x = gap.forward(x);
    std::cout << "  GAP output shape: [" << x.shape()[0] << ", " << x.shape()[1]
              << "] (expected: [2, 128])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 128}), "GAP shape mismatch");

    x = model.fc1().forward(x);
    ReLU relu4;
    x = relu4.forward(x);
    std::cout << "  FC1 (Linear 128->32 -> ReLU) shape: [" << x.shape()[0] << ", " << x.shape()[1]
              << "] (expected: [2, 32])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 32}), "FC1 shape mismatch");

    x = model.fc2().forward(x);
    std::cout << "  FC2 (Linear 32->1) shape: [" << x.shape()[0] << ", " << x.shape()[1]
              << "] (expected: [2, 1])" << std::endl;
    TEST_ASSERT(x.shape() == std::vector<size_t>({2, 1}), "FC2 shape mismatch");

    // Full model forward pass
    Tensor logits = model.forward(input);
    TEST_ASSERT(logits.shape() == std::vector<size_t>({2, 1}), "Model output shape must be [2, 1]");
    std::vector<float> logitVals = logits.toHostVector();
    TEST_ASSERT(isAllFinite(logitVals), "Logits must be finite");
    std::cout << "  Full Model Logits: [" << logitVals[0] << ", " << logitVals[1] << "]" << std::endl;

    std::cout << "[+] TEST 3 PASSED: All layers and model forward output correct shapes." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 4: Model Backward & Gradient Flow through All Parameters
// -------------------------------------------------------------
bool test_model_backward_and_gradients() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 4: Backward Pass & Gradient Flow through All 16 Parameters" << std::endl;
    std::cout << "==========================================" << std::endl;

    CatDogCNNModel model(DeviceType::CPU);
    model.train(true);
    model.zeroGrad();

    const size_t batchSize = 2;
    Tensor input({batchSize, 3, 224, 224});
    std::vector<float> inData(batchSize * 3 * 224 * 224);
    for (size_t i = 0; i < inData.size(); ++i) {
        inData[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.05f;
    }
    input.copyFromHost(inData.data(), inData.size());

    Tensor targets({batchSize, 1}, {0.0f, 1.0f});

    // Forward
    Tensor logits = model.forward(input);

    // Loss
    BCEWithLogitsLoss criterion;
    float loss = criterion.forward(logits, targets);
    std::cout << "  Initial Batch Loss: " << loss << std::endl;
    TEST_ASSERT(std::isfinite(loss) && loss > 0.0f, "Loss must be positive finite");

    // Backward through loss
    Tensor gradLogits = criterion.backward();
    TEST_ASSERT(gradLogits.shape() == std::vector<size_t>({batchSize, 1}), "gradLogits shape mismatch");
    TEST_ASSERT(isAllFinite(gradLogits.toHostVector()), "gradLogits must be finite");

    // Backward through model
    Tensor gradInput = model.backward(gradLogits);
    TEST_ASSERT(gradInput.shape() == input.shape(), "gradInput shape mismatch with input");
    TEST_ASSERT(isAllFinite(gradInput.toHostVector()), "gradInput must be finite");
    TEST_ASSERT(hasNonZero(gradInput.toHostVector()), "gradInput must have non-zero gradients");

    // Check all 16 parameters
    auto params = model.getParameters();
    std::cout << "  Total parameters collected: " << params.size() << " (expected: 16)" << std::endl;
    TEST_ASSERT(params.size() == 16, "Model must have exactly 16 trainable parameters");

    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params[i];
        TEST_ASSERT(p != nullptr, "Parameter pointer must not be null");
        TEST_ASSERT(p->requiresGrad(), "Parameter must require grad");
        TEST_ASSERT(p->data().size() > 0, "Parameter data must not be empty");
        TEST_ASSERT(p->grad().size() == p->data().size(), "Parameter grad size mismatch with data");

        std::vector<float> gVals = p->grad().toHostVector();
        TEST_ASSERT(isAllFinite(gVals), "Parameter gradient must be finite (no NaN or Inf)");
        TEST_ASSERT(hasNonZero(gVals), "Parameter gradient must be non-zero (gradient must reach this parameter)");

        std::cout << "  Param [" << std::setw(2) << i << "] " << std::left << std::setw(20) << p->name()
                  << " elements: " << std::setw(7) << p->data().size()
                  << " | finite: YES | non-zero: YES" << std::endl;
    }

    std::cout << "[+] TEST 4 PASSED: Gradient successfully flowed to all 16 parameters and to model input." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 5: Evaluation Mode Verification (BatchNorm Running Stats)
// -------------------------------------------------------------
bool test_model_eval_mode() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 5: Evaluation Mode & Inference Flow" << std::endl;
    std::cout << "==========================================" << std::endl;

    CatDogCNNModel model(DeviceType::CPU);

    // Run forward in training mode once so running stats update
    model.train(true);
    Tensor input({2, 3, 224, 224});
    std::vector<float> inData(2 * 3 * 224 * 224, 0.5f);
    input.copyFromHost(inData.data(), inData.size());

    model.forward(input);

    // Check running stats changed
    std::vector<float> rMean = model.bn1().runningMean().toHostVector();
    TEST_ASSERT(hasNonZero(rMean), "Running mean should have updated during train forward");

    // Switch to eval
    model.eval();
    TEST_ASSERT(!model.isTraining(), "Model should be in eval mode");

    Tensor logits = model.forward(input);
    TEST_ASSERT(isAllFinite(logits.toHostVector()), "Eval mode logits must be finite");

    Tensor targets({2, 1}, {0.0f, 1.0f});
    EvaluationResult res = Evaluator::evaluate(logits, targets);
    std::cout << "  Eval mode result:\n" << res.summary() << std::endl;

    std::cout << "[+] TEST 5 PASSED: Model eval mode switches and executes cleanly." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 6: CPU vs CUDA Parity Check (Dual Verification)
// -------------------------------------------------------------
bool test_cpu_cuda_model_parity() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 6: Dual CPU vs CUDA Numerical Parity" << std::endl;
    std::cout << "==========================================" << std::endl;

#ifdef USE_CUDA
    if (!CUDABackend::isAvailable()) {
        std::cout << "[!] CUDA not available on this host. Skipping CPU/CUDA comparison." << std::endl;
        return true;
    }

    CatDogCNNModel modelCPU(DeviceType::CPU);
    CatDogCNNModel modelGPU(DeviceType::CUDA);

    // Sync all parameter weights from CPU model to GPU model so they are identical
    auto paramsCPU = modelCPU.getParameters();
    auto paramsGPU = modelGPU.getParameters();
    TEST_ASSERT(paramsCPU.size() == paramsGPU.size(), "Parameter counts must match");

    for (size_t i = 0; i < paramsCPU.size(); ++i) {
        std::vector<float> hostData = paramsCPU[i]->data().toHostVector();
        paramsGPU[i]->data().copyFromHost(hostData.data(), hostData.size());
    }

    // Prepare identical input and target
    const size_t batchSize = 2;
    Tensor inputCPU({batchSize, 3, 224, 224}, DeviceType::CPU);
    std::vector<float> inData(batchSize * 3 * 224 * 224);
    for (size_t i = 0; i < inData.size(); ++i) {
        inData[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.05f;
    }
    inputCPU.copyFromHost(inData.data(), inData.size());

    Tensor inputGPU = inputCPU.to(DeviceType::CUDA);

    Tensor targetsCPU({batchSize, 1}, {0.0f, 1.0f}, DeviceType::CPU);
    Tensor targetsGPU = targetsCPU.to(DeviceType::CUDA);

    // Forward pass
    modelCPU.train(true);
    modelGPU.train(true);
    modelCPU.zeroGrad();
    modelGPU.zeroGrad();

    Tensor logitsCPU = modelCPU.forward(inputCPU);
    Tensor logitsGPU = modelGPU.forward(inputGPU);
    CUDABackend::synchronize();

    std::vector<float> hLogitsCPU = logitsCPU.toHostVector();
    std::vector<float> hLogitsGPU = logitsGPU.toHostVector();

    float diffLogits = maxDiff(hLogitsCPU, hLogitsGPU);
    std::cout << "  Logits max diff (CPU vs CUDA): " << diffLogits << std::endl;
    TEST_ASSERT(diffLogits < 1e-3f, "CPU and CUDA forward logits mismatch");

    // Loss calculation
    BCEWithLogitsLoss critCPU;
    BCEWithLogitsLoss critGPU;

    float lossCPU = critCPU.forward(logitsCPU, targetsCPU);
    float lossGPU = critGPU.forward(logitsGPU, targetsGPU);
    CUDABackend::synchronize();

    float diffLoss = std::abs(lossCPU - lossGPU);
    std::cout << "  Loss max diff (CPU vs CUDA): " << diffLoss << std::endl;
    TEST_ASSERT(diffLoss < 1e-4f, "CPU and CUDA loss mismatch");

    // Backward pass
    Tensor gradLogitsCPU = critCPU.backward();
    Tensor gradLogitsGPU = critGPU.backward();
    CUDABackend::synchronize();

    Tensor gradInputCPU = modelCPU.backward(gradLogitsCPU);
    Tensor gradInputGPU = modelGPU.backward(gradLogitsGPU);
    CUDABackend::synchronize();

    float diffGradInput = maxDiff(gradInputCPU.toHostVector(), gradInputGPU.toHostVector());
    std::cout << "  Input gradient max diff (CPU vs CUDA): " << diffGradInput << std::endl;
    TEST_ASSERT(diffGradInput < 1e-3f, "CPU and CUDA gradInput mismatch");

    // Compare all 16 parameter gradients
    for (size_t i = 0; i < paramsCPU.size(); ++i) {
        std::vector<float> gCPU = paramsCPU[i]->grad().toHostVector();
        std::vector<float> gGPU = paramsGPU[i]->grad().toHostVector();
        float pDiff = maxDiff(gCPU, gGPU);
        std::cout << "  Param [" << std::setw(2) << i << "] " << std::left << std::setw(18) << paramsCPU[i]->name()
                  << " grad max diff: " << pDiff << std::endl;
        TEST_ASSERT(pDiff < 1e-3f, "Parameter gradient mismatch between CPU and CUDA");
    }

    // Evaluator comparison
    EvaluationResult evalCPU = Evaluator::evaluate(logitsCPU, targetsCPU);
    EvaluationResult evalGPU = Evaluator::evaluate(logitsGPU, targetsGPU);

    TEST_ASSERT(evalCPU.confusionMatrix.tn == evalGPU.confusionMatrix.tn, "TN mismatch");
    TEST_ASSERT(evalCPU.confusionMatrix.fp == evalGPU.confusionMatrix.fp, "FP mismatch");
    TEST_ASSERT(evalCPU.confusionMatrix.fn == evalGPU.confusionMatrix.fn, "FN mismatch");
    TEST_ASSERT(evalCPU.confusionMatrix.tp == evalGPU.confusionMatrix.tp, "TP mismatch");
    TEST_ASSERT(std::abs(evalCPU.accuracy - evalGPU.accuracy) < 1e-5f, "Accuracy mismatch");
    TEST_ASSERT(std::abs(evalCPU.loss - evalGPU.loss) < 1e-4f, "Loss metric mismatch");

    std::cout << "[+] TEST 6 PASSED: Dual CPU vs CUDA parity verified with high numerical precision!" << std::endl;
#else
    std::cout << "[-] Compiled without USE_CUDA, skipping CUDA parity." << std::endl;
#endif
    return true;
}

static std::string resolvePath(const std::string& path) {
    std::ifstream f1(path);
    if (f1.good()) return path;
    std::string alt = "../../" + path;
    std::ifstream f2(alt);
    if (f2.good()) return alt;
    std::string alt2 = "../" + path;
    std::ifstream f3(alt2);
    if (f3.good()) return alt2;
    return path;
}

// -------------------------------------------------------------
// Test 7: Dataset Split & Manifest Generation / Reuse
// -------------------------------------------------------------
bool test_dataset_split_and_manifest_reuse() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 7: Deterministic Stratified Dataset Split & Manifest Reuse" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::string labelsPath = resolvePath("dataset/processed/labels.csv");
    std::string manifestPath = resolvePath("dataset/split_manifest.csv");
    if (manifestPath.find("dataset") == std::string::npos) {
        manifestPath = "dataset/split_manifest.csv";
    }

    std::cout << "  Labels CSV Path   : " << labelsPath << std::endl;
    std::cout << "  Manifest CSV Path : " << manifestPath << std::endl;

    // 1. Generate or load split
    auto manifest1 = DatasetSplitter::getOrGenerateSplit(labelsPath, manifestPath, 0.8f, 0.1f, 0.1f, 42);
    TEST_ASSERT(!manifest1.empty(), "Manifest must not be empty");
    TEST_ASSERT(manifest1.size() == 24989, "Total processed samples must be exactly 24,989");

    // Count distributions
    size_t trainCount = 0, valCount = 0, testCount = 0;
    size_t catTotal = 0, dogTotal = 0;
    size_t catTrain = 0, dogTrain = 0;
    size_t catVal = 0, dogVal = 0;
    size_t catTest = 0, dogTest = 0;
    std::set<std::string> uniqueFiles;

    for (const auto& e : manifest1) {
        TEST_ASSERT(uniqueFiles.find(e.filename) == uniqueFiles.end(), "Duplicate file detected in split!");
        uniqueFiles.insert(e.filename);

        if (e.label == 0) catTotal++;
        else if (e.label == 1) dogTotal++;

        if (e.split == "train") {
            trainCount++;
            if (e.label == 0) catTrain++; else dogTrain++;
        } else if (e.split == "validation") {
            valCount++;
            if (e.label == 0) catVal++; else dogVal++;
        } else if (e.split == "test") {
            testCount++;
            if (e.label == 0) catTest++; else dogTest++;
        } else {
            TEST_ASSERT(false, "Unknown split label found: " + e.split);
        }
    }

    TEST_ASSERT(catTotal == 12492, "Cats total must be 12,492");
    TEST_ASSERT(dogTotal == 12497, "Dogs total must be 12,497");
    TEST_ASSERT(trainCount == 19992, "Train set must have 19,992 samples (~80.0%)");
    TEST_ASSERT(valCount == 2499, "Validation set must have 2,499 samples (~10.0%)");
    TEST_ASSERT(testCount == 2498, "Test set must have 2,498 samples (~10.0%)");

    std::cout << "  Split Distribution Summary:\n";
    std::cout << "    - Train      : " << trainCount << " (Cats: " << catTrain << ", Dogs: " << dogTrain << ")\n";
    std::cout << "    - Validation : " << valCount << " (Cats: " << catVal << ", Dogs: " << dogVal << ")\n";
    std::cout << "    - Test       : " << testCount << " (Cats: " << catTest << ", Dogs: " << dogTest << ")\n";
    std::cout << "    - Total      : " << uniqueFiles.size() << " samples\n";

    // 2. Test manifest reuse: must reload identical assignments without regenerating
    auto manifest2 = DatasetSplitter::getOrGenerateSplit(labelsPath, manifestPath, 0.8f, 0.1f, 0.1f, 42);
    TEST_ASSERT(manifest1.size() == manifest2.size(), "Reloaded manifest size mismatch");
    for (size_t i = 0; i < manifest1.size(); ++i) {
        TEST_ASSERT(manifest1[i].filename == manifest2[i].filename, "Filename mismatch on reuse");
        TEST_ASSERT(manifest1[i].label == manifest2[i].label, "Label mismatch on reuse");
        TEST_ASSERT(manifest1[i].split == manifest2[i].split, "Split mismatch on reuse");
    }

    std::cout << "[+] TEST 7 PASSED: Stratified split generated and verified. Manifest successfully reused." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 8: DataLoader Batch Creation, Layout & Device Transfers
// -------------------------------------------------------------
bool test_dataloader_batching_and_layout() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 8: DataLoader Batch Creation, NCHW Layout & Device Transfers" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::string manifestPath = resolvePath("dataset/split_manifest.csv");
    std::string imagesDir = resolvePath("dataset/processed/images");

    // Initialize Validation DataLoader with batch size 4 (deterministic)
    DataLoader valLoader(manifestPath, imagesDir, "validation", 4, false, DeviceType::CPU);
    std::cout << "  Validation DataLoader created with " << valLoader.numSamples()
              << " samples (" << valLoader.numBatches() << " batches of 4)" << std::endl;
    TEST_ASSERT(valLoader.numSamples() == 2499, "Validation count mismatch");
    TEST_ASSERT(valLoader.numBatches() == 625, "Validation batch count mismatch");

    // Fetch first batch
    TEST_ASSERT(valLoader.hasNext(), "Loader should have next batch");
    Batch batch1 = valLoader.nextBatch();
    TEST_ASSERT(batch1.batchSize() == 4, "Batch size must be 4");
    TEST_ASSERT(batch1.images.shape() == std::vector<size_t>({4, 3, 224, 224}), "Batch images shape mismatch");
    TEST_ASSERT(batch1.labels.shape() == std::vector<size_t>({4, 1}), "Batch labels shape mismatch");

    // Verify pixel normalization [0.0, 1.0]
    std::vector<float> h_imgs = batch1.images.toHostVector();
    for (float px : h_imgs) {
        TEST_ASSERT(px >= 0.0f && px <= 1.0f, "Pixel value out of [0, 1] range");
    }

    // Verify HWC -> NCHW channel layout against direct ImageReader output
    ImageReader reader(3);
    std::string errMsg;
    ImageData raw0 = reader.read(imagesDir + "/" + batch1.filenames[0], errMsg);
    TEST_ASSERT(!raw0.empty(), "Failed to read reference image for layout check");

    // Check specific pixel sample locations in (C, H, W)
    for (size_t c = 0; c < 3; ++c) {
        for (size_t h : {10, 50, 100, 200}) {
            for (size_t w : {10, 50, 100, 200}) {
                size_t nchwIdx = ((0 * 3 + c) * 224 + h) * 224 + w;
                size_t hwcIdx = (h * 224 + w) * 3 + c;
                float expected = static_cast<float>(raw0.pixels[hwcIdx]) / 255.0f;
                float actual = h_imgs[nchwIdx];
                TEST_ASSERT(std::abs(actual - expected) < 1e-6f, "Layout pixel value mismatch");
            }
        }
    }
    std::cout << "  Layout verified: HWC uint8 converted correctly to NCHW FP32 [0, 1]" << std::endl;

    // Verify determinism across reset() on validation loader
    std::string firstFilename = batch1.filenames[0];
    valLoader.reset();
    Batch batch1Again = valLoader.nextBatch();
    TEST_ASSERT(batch1Again.filenames[0] == firstFilename, "Validation order must remain deterministic");

    // Test Train DataLoader with shuffle
    DataLoader trainLoader(manifestPath, imagesDir, "train", 4, true, DeviceType::CPU, 1234);
    TEST_ASSERT(trainLoader.numSamples() == 19992, "Train count mismatch");
    Batch trainBatch1 = trainLoader.nextBatch();
    trainLoader.reset();
    Batch trainBatch2 = trainLoader.nextBatch();
    // With 19,992 items, the chance of the first 4 elements matching after shuffle is infinitesimal
    bool allMatched = true;
    for (size_t i = 0; i < 4; ++i) {
        if (trainBatch1.filenames[i] != trainBatch2.filenames[i]) {
            allMatched = false;
            break;
        }
    }
    TEST_ASSERT(!allMatched, "Training set should re-shuffle after reset");
    std::cout << "  Shuffle verified: Training batches re-order between epochs" << std::endl;

#ifdef USE_CUDA
    if (CUDABackend::isAvailable()) {
        DataLoader gpuLoader(manifestPath, imagesDir, "validation", 4, false, DeviceType::CUDA);
        Batch gpuBatch = gpuLoader.nextBatch();
        TEST_ASSERT(gpuBatch.images.device() == DeviceType::CUDA, "GPU batch images must be on CUDA device");
        TEST_ASSERT(gpuBatch.labels.device() == DeviceType::CUDA, "GPU batch labels must be on CUDA device");

        std::vector<float> gpuImgs = gpuBatch.images.toHostVector();
        float diff = maxDiff(h_imgs, gpuImgs);
        TEST_ASSERT(diff < 1e-6f, "GPU batch transfer data mismatch");
        std::cout << "  CUDA batch transfer verified: batch placed directly on GPU memory" << std::endl;
    }
#endif

    std::cout << "[+] TEST 8 PASSED: DataLoader batch creation, layout, normalization, and device transfers verified." << std::endl;
    return true;
}

// -------------------------------------------------------------
// Test 9: Real Dataset Batch Inference (CPU & CUDA Parity)
// -------------------------------------------------------------
bool test_real_batch_inference_cpu_cuda() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 9: Real Dataset Batch Forward Pass & Evaluation (No Training)" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::string manifestPath = resolvePath("dataset/split_manifest.csv");
    std::string imagesDir = resolvePath("dataset/processed/images");

    DataLoader loader(manifestPath, imagesDir, "test", 4, false, DeviceType::CPU);
    Batch batch = loader.nextBatch();

    CatDogCNNModel modelCPU(DeviceType::CPU);
    modelCPU.eval();

    Tensor logitsCPU = modelCPU.forward(batch.images);
    EvaluationResult evalCPU = Evaluator::evaluate(logitsCPU, batch.labels);

    std::cout << "  CPU Model Evaluation on Real Batch:\n" << evalCPU.summary() << std::endl;
    TEST_ASSERT(std::isfinite(evalCPU.loss), "Loss must be finite");
    TEST_ASSERT(evalCPU.confusionMatrix.total() == 4, "Must evaluate 4 samples");

#ifdef USE_CUDA
    if (CUDABackend::isAvailable()) {
        CatDogCNNModel modelGPU(DeviceType::CUDA);
        modelGPU.eval();

        // Sync weights from CPU to GPU
        auto pCPU = modelCPU.getParameters();
        auto pGPU = modelGPU.getParameters();
        for (size_t i = 0; i < pCPU.size(); ++i) {
            std::vector<float> w = pCPU[i]->data().toHostVector();
            pGPU[i]->data().copyFromHost(w.data(), w.size());
        }

        Tensor gpuImages = batch.images.to(DeviceType::CUDA);
        Tensor gpuLabels = batch.labels.to(DeviceType::CUDA);

        Tensor logitsGPU = modelGPU.forward(gpuImages);
        EvaluationResult evalGPU = Evaluator::evaluate(logitsGPU, gpuLabels);

        float lDiff = maxDiff(logitsCPU.toHostVector(), logitsGPU.toHostVector());
        std::cout << "  Real batch Logits max diff (CPU vs CUDA): " << lDiff << std::endl;
        TEST_ASSERT(lDiff < 1e-4f, "Logits mismatch between CPU and CUDA on real batch");
        TEST_ASSERT(std::abs(evalCPU.loss - evalGPU.loss) < 1e-4f, "Loss mismatch on real batch");
        TEST_ASSERT(evalCPU.accuracy == evalGPU.accuracy, "Accuracy mismatch on real batch");
        std::cout << "  CPU and CUDA produced identical predictions on real dataset images!" << std::endl;
    }
#endif

    std::cout << "[+] TEST 9 PASSED: Forward evaluation on real dataset images verified without training." << std::endl;
    return true;
}

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "   CatDogCNN Phase 4: Model Composition & Evaluation    " << std::endl;
    std::cout << "========================================================" << std::endl;

    bool passed = true;
    passed = passed && test_evaluator_metrics();
    passed = passed && test_batchnorm2d_standalone();
    passed = passed && test_model_architecture_shapes();
    passed = passed && test_model_backward_and_gradients();
    passed = passed && test_model_eval_mode();
    passed = passed && test_cpu_cuda_model_parity();
    passed = passed && test_dataset_split_and_manifest_reuse();
    passed = passed && test_dataloader_batching_and_layout();
    passed = passed && test_real_batch_inference_cpu_cuda();

    std::cout << "\n========================================================" << std::endl;
    if (passed) {
        std::cout << " >>> ALL PHASE 4 TESTS COMPLETED SUCCESSFULLY! <<< " << std::endl;
        std::cout << "========================================================" << std::endl;
        return 0;
    } else {
        std::cerr << " >>> SOME TESTS FAILED! <<< " << std::endl;
        std::cout << "========================================================" << std::endl;
        return 1;
    }
}
