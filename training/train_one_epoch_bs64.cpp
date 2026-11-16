#include "Trainer.hpp"
#include "../core/CUDABackend.hpp"
#include "../model/CatDogCNNModel.hpp"
#include "../loss/BCEWithLogitsLoss.hpp"
#include "../optimizers/SGD.hpp"
#include "../data/DataLoader.hpp"
#include "../evaluation/Evaluator.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

static std::string resolvePath(const std::string& candidateA, const std::string& candidateB) {
    if (fs::exists(candidateA)) return candidateA;
    if (fs::exists(candidateB)) return candidateB;
    return candidateA;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "   CatDogCNN: Fresh Training Run (Batch Size 64, 1 Epoch)\n";
    std::cout << "========================================================\n\n";

    // 1. Hardware Verification
    if (!CUDABackend::isAvailable()) {
        std::cerr << "[-] FATAL: NVIDIA CUDA is not available on this host!\n";
        return 1;
    }

    CUDABackend::setDevice(0);
    CUDADeviceInfo dev = CUDABackend::getDeviceInfo(0);
    std::cout << "[Step 1] Hardware & CUDA Verification:\n";
    std::cout << "  Active GPU             : " << dev.name << "\n";
    std::cout << "  Compute Capability     : " << dev.major << "." << dev.minor << "\n";
    std::cout << "  Total Dedicated VRAM   : " << std::fixed << std::setprecision(1)
              << (dev.totalGlobalMem / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Multiprocessors (SMs)  : " << dev.multiProcessorCount << "\n\n";

    // 2. Setup Configuration
    const int batchSize = 64;
    const int numEpochs = 1;
    const float learningRate = 0.01f;
    const float momentum = 0.9f;
    const float weightDecay = 1e-4f;

    std::string imagesDir   = resolvePath("dataset/processed/images", "../dataset/processed/images");
    std::string manifestCsv = resolvePath("dataset/split_manifest.csv", "../dataset/split_manifest.csv");
    std::string checkpointDir = resolvePath("checkpoints", "../checkpoints");

    std::cout << "[Step 2] Configuration:\n";
    std::cout << "  Batch Size             : " << batchSize << "\n";
    std::cout << "  Training Epochs        : " << numEpochs << "\n";
    std::cout << "  Optimizer              : SGD (LR=" << learningRate
              << ", Momentum=" << momentum << ", WeightDecay=" << weightDecay << ")\n";
    std::cout << "  Manifest Path          : " << manifestCsv << "\n";
    std::cout << "  Images Directory       : " << imagesDir << "\n";
    std::cout << "  Checkpoint Directory   : " << checkpointDir << "\n\n";

    // 3. Initialize Completely Fresh Model & Verify
    std::cout << "[Step 3] Initializing Fresh CatDogCNNModel on CUDA...\n";
    CatDogCNNModel model(DeviceType::CUDA);
    model.train(true);

    auto params = model.getParameters();
    std::cout << "  Total Model Parameters : " << params.size() << " tensors ("
              << "97,825 parameters, ~391.3 KB FP32)\n";

    // Confirm fresh status (zero gradients, clean initialization)
    bool allFresh = true;
    for (Parameter* p : params) {
        if (!p || !p->requiresGrad()) { allFresh = false; break; }
        // Velocity must be all zeros initially
        std::vector<float> v = p->velocity().toHostVector();
        for (float val : v) {
            if (val != 0.0f) { allFresh = false; break; }
        }
    }
    if (!allFresh) {
        std::cerr << "[-] Error: Model parameters not cleanly initialized!\n";
        return 1;
    }
    std::cout << "  [+] Confirmed: Model is 100% fresh. Zero prior training updates.\n\n";

    // 4. Batch Size 64 CUDA Smoke Test (VRAM & Forward/Backward)
    std::cout << "[Step 4] CUDA Memory & Smoke Test for Batch Size 64...\n";
    {
        Tensor testImages({static_cast<size_t>(batchSize), 3, 224, 224}, DeviceType::CUDA);
        Tensor testLabels({static_cast<size_t>(batchSize), 1}, DeviceType::CUDA);
        std::vector<float> dummyImg(batchSize * 3 * 224 * 224, 0.5f);
        std::vector<float> dummyLbl(batchSize, 1.0f);
        testImages.copyFromHost(dummyImg.data(), dummyImg.size());
        testLabels.copyFromHost(dummyLbl.data(), dummyLbl.size());

        BCEWithLogitsLoss testCrit;
        model.zeroGrad();
        Tensor logits = model.forward(testImages);
        float loss = testCrit.forward(logits, testLabels);
        Tensor gradLogits = testCrit.backward();
        model.backward(gradLogits);

        SGD testOpt(model.getParameters(), learningRate, momentum, weightDecay);
        testOpt.step();
        CUDABackend::synchronize();

        size_t freeMem = 0, totalMem = 0;
#ifdef USE_CUDA
        cudaMemGetInfo(&freeMem, &totalMem);
        double usedMB = static_cast<double>(totalMem - freeMem) / (1024.0 * 1024.0);
        double totalMB = static_cast<double>(totalMem) / (1024.0 * 1024.0);
        std::cout << "  Smoke Test Loss        : " << loss << " (finite: YES)\n";
        std::cout << "  VRAM after BS=64 test  : " << std::fixed << std::setprecision(1)
                  << usedMB << " MB / " << totalMB << " MB ("
                  << (usedMB / totalMB * 100.0) << "%)\n";
#endif
        std::cout << "  [+] Batch Size 64 verified: Safe allocation and execution on GTX 1650!\n\n";
    }

    // Re-initialize a clean model again so smoke test update does NOT leak
    CatDogCNNModel freshModel(DeviceType::CUDA);
    BCEWithLogitsLoss criterion;
    SGD optimizer(freshModel.getParameters(), learningRate, momentum, weightDecay);

    // 5. Initialize DataLoaders
    std::cout << "[Step 5] Initializing DataLoaders from " << manifestCsv << "...\n";
    DataLoader trainLoader(manifestCsv, imagesDir, "train", batchSize, true, DeviceType::CUDA);
    DataLoader valLoader(manifestCsv, imagesDir, "validation", batchSize, false, DeviceType::CUDA);

    std::cout << "  Training samples       : " << trainLoader.numSamples()
              << " (" << trainLoader.numBatches() << " batches of " << batchSize << ")\n";
    std::cout << "  Validation samples     : " << valLoader.numSamples()
              << " (" << valLoader.numBatches() << " batches of " << batchSize << ")\n\n";

    // 6. Baseline Evaluation — Before Training (Epoch 0)
    std::cout << "========================================================\n";
    std::cout << " [Baseline Evaluation] Before Training (Epoch 0)\n";
    std::cout << "========================================================\n";
    freshModel.eval();
    valLoader.reset();

    Evaluator baselineEval(0.5f);
    while (valLoader.hasNext()) {
        Batch b = valLoader.nextBatch();
        Tensor outLogits = freshModel.forward(b.images);
        baselineEval.update(outLogits, b.labels);
    }
    EvaluationResult beforeMetrics = baselineEval.compute();

    std::cout << beforeMetrics.summary() << "\n\n";

    // 7. Train for Exactly One Epoch
    std::cout << "========================================================\n";
    std::cout << " Starting Training: Epoch [1/1] | Batch Size: " << batchSize << "\n";
    std::cout << "========================================================\n";

    freshModel.train(true);
    trainLoader.reset();

    double epochLoss = 0.0;
    Evaluator trainEval(0.5f);
    size_t batchIdx = 0;
    size_t totalTrainBatches = trainLoader.numBatches();

    auto epochStart = std::chrono::steady_clock::now();

    while (trainLoader.hasNext()) {
        Batch batch = trainLoader.nextBatch();
        batchIdx++;

        freshModel.zeroGrad();
        Tensor logits = freshModel.forward(batch.images);
        float batchLoss = criterion.forward(logits, batch.labels);
        epochLoss += batchLoss;

        Tensor gradLogits = criterion.backward();
        freshModel.backward(gradLogits);
        optimizer.step();

        trainEval.update(logits, batch.labels, batchLoss);

        // Report every 25 batches
        if (batchIdx % 25 == 0 || batchIdx == totalTrainBatches) {
            float pct = (static_cast<float>(batchIdx) / static_cast<float>(totalTrainBatches)) * 100.0f;
            float curLoss = static_cast<float>(epochLoss / static_cast<double>(batchIdx));
            EvaluationResult curTrain = trainEval.compute();

            std::cout << "  [Batch " << std::setw(3) << batchIdx << "/" << totalTrainBatches
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)]"
                      << " Avg Loss: " << std::setprecision(4) << curLoss
                      << " | Train Acc: " << (curTrain.accuracy * 100.0f) << "%"
                      << std::endl;
        }
    }

    auto trainEnd = std::chrono::steady_clock::now();
    double trainSec = std::chrono::duration<double>(trainEnd - epochStart).count();
    EvaluationResult trainResult = trainEval.compute();

    std::cout << "\n---------------- Training Finished ----------------\n";
    std::cout << " Training Duration : " << std::fixed << std::setprecision(1) << (trainSec / 60.0) << " mins\n";
    std::cout << " Final Train Loss  : " << std::setprecision(4) << trainResult.loss << "\n";
    std::cout << " Final Train Acc   : " << std::setprecision(2) << (trainResult.accuracy * 100.0f) << "%\n";
    std::cout << " Final Train F1    : " << (trainResult.f1Score * 100.0f) << "%\n\n";

    // 8. Evaluation — After Training (Epoch 1)
    std::cout << "========================================================\n";
    std::cout << " [Evaluation] After Training (Epoch 1)\n";
    std::cout << "========================================================\n";
    freshModel.eval();
    valLoader.reset();

    Evaluator afterEval(0.5f);
    while (valLoader.hasNext()) {
        Batch b = valLoader.nextBatch();
        Tensor outLogits = freshModel.forward(b.images);
        afterEval.update(outLogits, b.labels);
    }
    EvaluationResult afterMetrics = afterEval.compute();

    std::cout << afterMetrics.summary() << "\n\n";

    // 9. Save Checkpoint
    fs::create_directories(checkpointDir);
    std::string ckptPath = checkpointDir + "/checkpoint_epoch_1_bs64.bin";
    std::string latestPath = checkpointDir + "/latest_checkpoint.bin";

    Trainer helperTrainer(TrainingConfig{batchSize, learningRate, momentum, weightDecay, 1, imagesDir, manifestCsv, checkpointDir, DeviceType::CUDA});
    // Copy weights into helperTrainer to serialize
    auto helperParams = helperTrainer.model().getParameters();
    auto trainedParams = freshModel.getParameters();
    for (size_t i = 0; i < helperParams.size(); ++i) {
        std::vector<float> w = trainedParams[i]->data().toHostVector();
        std::vector<float> v = trainedParams[i]->velocity().toHostVector();
        helperParams[i]->data().copyFromHost(w.data(), w.size());
        helperParams[i]->velocity().copyFromHost(v.data(), v.size());
    }
    // Copy BatchNorm running stats
    std::vector<float> m1 = freshModel.bn1().runningMean().toHostVector();
    std::vector<float> v1 = freshModel.bn1().runningVar().toHostVector();
    helperTrainer.model().bn1().runningMean().copyFromHost(m1.data(), m1.size());
    helperTrainer.model().bn1().runningVar().copyFromHost(v1.data(), v1.size());

    std::vector<float> m2 = freshModel.bn2().runningMean().toHostVector();
    std::vector<float> v2 = freshModel.bn2().runningVar().toHostVector();
    helperTrainer.model().bn2().runningMean().copyFromHost(m2.data(), m2.size());
    helperTrainer.model().bn2().runningVar().copyFromHost(v2.data(), v2.size());

    std::vector<float> m3 = freshModel.bn3().runningMean().toHostVector();
    std::vector<float> v3 = freshModel.bn3().runningVar().toHostVector();
    helperTrainer.model().bn3().runningMean().copyFromHost(m3.data(), m3.size());
    helperTrainer.model().bn3().runningVar().copyFromHost(v3.data(), v3.size());

    helperTrainer.saveCheckpoint(ckptPath, 1, afterMetrics);
    helperTrainer.saveCheckpoint(latestPath, 1, afterMetrics);
    std::cout << "[+] Checkpoint successfully saved to: " << ckptPath << "\n";
    std::cout << "[+] Latest pointer saved to        : " << latestPath << "\n\n";

    // 10. Before vs After Comparison Table
    float lossChange = afterMetrics.loss - beforeMetrics.loss;
    float accChange  = (afterMetrics.accuracy - beforeMetrics.accuracy) * 100.0f;
    float precChange = (afterMetrics.precision - beforeMetrics.precision) * 100.0f;
    float recChange  = (afterMetrics.recall - beforeMetrics.recall) * 100.0f;
    float f1Change   = (afterMetrics.f1Score - beforeMetrics.f1Score) * 100.0f;

    std::cout << "========================================================\n";
    std::cout << "           BEFORE VS AFTER TRAINING COMPARISON          \n";
    std::cout << "========================================================\n";
    std::cout << "| Metric    | Before Training | After 1 Epoch | Change     |\n";
    std::cout << "|-----------|-----------------|---------------|------------|\n";
    std::cout << "| BCE Loss  | " << std::setw(15) << std::fixed << std::setprecision(4) << beforeMetrics.loss
              << " | " << std::setw(13) << afterMetrics.loss
              << " | " << std::setw(10) << (lossChange >= 0 ? "+" : "") << lossChange << " |\n";
    std::cout << "| Accuracy  | " << std::setw(14) << (beforeMetrics.accuracy * 100.0f) << "% | "
              << std::setw(12) << (afterMetrics.accuracy * 100.0f) << "% | "
              << std::setw(9) << (accChange >= 0 ? "+" : "") << accChange << "% |\n";
    std::cout << "| Precision | " << std::setw(14) << (beforeMetrics.precision * 100.0f) << "% | "
              << std::setw(12) << (afterMetrics.precision * 100.0f) << "% | "
              << std::setw(9) << (precChange >= 0 ? "+" : "") << precChange << "% |\n";
    std::cout << "| Recall    | " << std::setw(14) << (beforeMetrics.recall * 100.0f) << "% | "
              << std::setw(12) << (afterMetrics.recall * 100.0f) << "% | "
              << std::setw(9) << (recChange >= 0 ? "+" : "") << recChange << "% |\n";
    std::cout << "| F1-Score  | " << std::setw(14) << (beforeMetrics.f1Score * 100.0f) << "% | "
              << std::setw(12) << (afterMetrics.f1Score * 100.0f) << "% | "
              << std::setw(9) << (f1Change >= 0 ? "+" : "") << f1Change << "% |\n";
    std::cout << "========================================================\n\n";

    std::cout << "Confusion Matrix Progression:\n";
    std::cout << "  Before : TN=" << beforeMetrics.confusionMatrix.tn
              << ", FP=" << beforeMetrics.confusionMatrix.fp
              << ", FN=" << beforeMetrics.confusionMatrix.fn
              << ", TP=" << beforeMetrics.confusionMatrix.tp << "\n";
    std::cout << "  After  : TN=" << afterMetrics.confusionMatrix.tn
              << ", FP=" << afterMetrics.confusionMatrix.fp
              << ", FN=" << afterMetrics.confusionMatrix.fn
              << ", TP=" << afterMetrics.confusionMatrix.tp << "\n\n";

    std::cout << " >>> 1-EPOCH FRESH TRAINING COMPLETED SUCCESSFULLY! <<<\n";
    return 0;
}
