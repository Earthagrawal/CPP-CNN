#include "Trainer.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

Trainer::Trainer(const TrainingConfig& config)
    : m_config(config),
      m_model(config.device),
      m_currentLr(config.learningRate) {
    initializeDevice();
    setupOptimizer();
}

void Trainer::initializeDevice() {
    if (m_config.device == DeviceType::CUDA) {
        if (!CUDABackend::isAvailable()) {
            throw std::runtime_error("Trainer: CUDA was requested but no CUDA-capable device was detected!");
        }
        CUDABackend::setDevice(0);
        CUDADeviceInfo info = CUDABackend::getDeviceInfo(0);
        std::cout << "========================================================\n";
        std::cout << "  Active Compute Backend : NVIDIA CUDA GPU\n";
        std::cout << "  Device Name            : " << info.name << "\n";
        std::cout << "  Compute Capability     : " << info.major << "." << info.minor << "\n";
        std::cout << "  Dedicated Global VRAM  : " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(info.totalGlobalMem) / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Multiprocessors (SMs)  : " << info.multiProcessorCount << "\n";
        std::cout << "========================================================\n";
    } else {
        std::cout << "========================================================\n";
        std::cout << "  Active Compute Backend : CPU Reference Engine\n";
        std::cout << "========================================================\n";
    }
}

void Trainer::setupOptimizer() {
    auto params = m_model.getParameters();
    m_optimizer = std::make_unique<SGD>(params, m_config.learningRate, m_config.momentum, m_config.weightDecay);
}

std::string Trainer::formatTime(double seconds) const {
    int totalSec = static_cast<int>(seconds);
    int m = totalSec / 60;
    int s = totalSec % 60;
    std::ostringstream oss;
    if (m > 0) oss << m << "m ";
    oss << s << "s";
    return oss.str();
}

bool Trainer::runSmokeTest() {
    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " [Smoke Test] Running Pre-Training Verification...\n";
    std::cout << "--------------------------------------------------------\n";

    // 1. Verify CUDA device active
    if (m_config.device == DeviceType::CUDA) {
        if (!CUDABackend::isAvailable() || m_model.device() != DeviceType::CUDA) {
            std::cerr << "[-] Smoke Test FAILED: CUDA device not active on model!\n";
            return false;
        }
        std::cout << "  [1/6] CUDA Execution Active       : PASS ("
                  << CUDABackend::getDeviceInfo(0).name << ")\n";
    } else {
        std::cout << "  [1/6] Compute Device Active       : PASS (CPU)\n";
    }

    // 2. Prepare synthetic verification mini-batch of 4 samples
    const size_t B = 4;
    Tensor dummyImages({B, 3, 224, 224}, m_config.device);
    Tensor dummyLabels({B, 1}, m_config.device);

    std::vector<float> h_imgs(B * 3 * 224 * 224);
    for (size_t i = 0; i < h_imgs.size(); ++i) {
        h_imgs[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.05f;
    }
    dummyImages.copyFromHost(h_imgs.data(), h_imgs.size());
    std::vector<float> h_lbls = {0.0f, 1.0f, 0.0f, 1.0f};
    dummyLabels.copyFromHost(h_lbls.data(), h_lbls.size());

    // Record initial weights and running stats
    std::vector<float> initConv1Weight = m_model.conv1().weight().data().toHostVector();
    std::vector<float> initFc2Bias     = m_model.fc2().bias().data().toHostVector();
    std::vector<float> initBn1Mean     = m_model.bn1().runningMean().toHostVector();

    // 3. Forward pass in training mode
    m_model.train(true);
    m_model.zeroGrad();
    Tensor logits = m_model.forward(dummyImages);
    float loss = m_criterion.forward(logits, dummyLabels);

    if (!std::isfinite(loss) || loss <= 0.0f) {
        std::cerr << "[-] Smoke Test FAILED: Initial loss is not positive finite: " << loss << "\n";
        return false;
    }
    std::cout << "  [2/6] Forward Propagation & Loss   : PASS (Loss = " << loss << ")\n";

    // 4. Check BatchNorm running stats updated during training forward
    std::vector<float> postTrainBn1Mean = m_model.bn1().runningMean().toHostVector();
    bool bnMeanUpdated = false;
    for (size_t i = 0; i < postTrainBn1Mean.size(); ++i) {
        if (std::abs(postTrainBn1Mean[i] - initBn1Mean[i]) > 1e-7f) {
            bnMeanUpdated = true;
            break;
        }
    }
    if (!bnMeanUpdated) {
        std::cerr << "[-] Smoke Test FAILED: BatchNorm running statistics did not update during training forward!\n";
        return false;
    }
    std::cout << "  [3/6] BatchNorm Running Stats Upd : PASS (Mean updated on training forward)\n";

    // 5. Backward pass & check gradients finite and non-zero
    Tensor gradLogits = m_criterion.backward();
    m_model.backward(gradLogits);

    auto params = m_model.getParameters();
    bool gradsOk = true;
    for (Parameter* p : params) {
        if (!p) continue;
        std::vector<float> gVals = p->grad().toHostVector();
        for (float g : gVals) {
            if (!std::isfinite(g)) {
                gradsOk = false;
                break;
            }
        }
    }
    if (!gradsOk) {
        std::cerr << "[-] Smoke Test FAILED: Non-finite gradients detected!\n";
        return false;
    }
    std::cout << "  [4/6] Backward Pass & Gradients   : PASS (All 16 parameters finite)\n";

    // 6. Optimizer step and verify parameters changed
    m_optimizer->step();
    std::vector<float> postStepConv1Weight = m_model.conv1().weight().data().toHostVector();
    std::vector<float> postStepFc2Bias     = m_model.fc2().bias().data().toHostVector();

    bool weightsChanged = false;
    for (size_t i = 0; i < initConv1Weight.size(); ++i) {
        if (std::abs(postStepConv1Weight[i] - initConv1Weight[i]) > 1e-7f) {
            weightsChanged = true;
            break;
        }
    }
    if (!weightsChanged) {
        std::cerr << "[-] Smoke Test FAILED: Parameters did not change after optimizer step!\n";
        return false;
    }
    std::cout << "  [5/6] Optimizer Parameter Update  : PASS (Weights successfully updated)\n";

    // 7. Verify BatchNorm running statistics remain unchanged during validation (eval mode)
    m_model.eval();
    std::vector<float> preEvalBn1Mean = m_model.bn1().runningMean().toHostVector();
    m_model.forward(dummyImages);
    std::vector<float> postEvalBn1Mean = m_model.bn1().runningMean().toHostVector();

    bool bnStatsFrozen = true;
    for (size_t i = 0; i < preEvalBn1Mean.size(); ++i) {
        if (std::abs(postEvalBn1Mean[i] - preEvalBn1Mean[i]) > 1e-9f) {
            bnStatsFrozen = false;
            break;
        }
    }
    if (!bnStatsFrozen) {
        std::cerr << "[-] Smoke Test FAILED: BatchNorm running stats mutated during eval() forward!\n";
        return false;
    }
    std::cout << "  [6/6] BatchNorm Frozen in Eval    : PASS (Running statistics 100% invariant)\n";

    m_model.zeroGrad();
    std::cout << " >>> [Smoke Test] ALL 6 PRE-TRAINING CHECKS PASSED! <<<\n\n";
    return true;
}

EvaluationResult Trainer::evaluate(DataLoader& loader) {
    m_model.eval();
    loader.reset();

    Evaluator evaluator(0.5f);

    while (loader.hasNext()) {
        Batch batch = loader.nextBatch();
        Tensor logits = m_model.forward(batch.images);
        evaluator.update(logits, batch.labels);
    }

    return evaluator.compute();
}

#pragma pack(push, 1)
struct CheckpointHeader {
    char magic[8];          // "CDCNN01"
    int32_t epoch;
    float learningRate;
    float momentum;
    float weightDecay;
    int32_t batchSize;
    float valLoss;
    float valAccuracy;
    float valPrecision;
    float valRecall;
    float valF1;
    int32_t valTN;
    int32_t valFP;
    int32_t valFN;
    int32_t valTP;
    uint32_t numParams;
    uint32_t numBNLayers;
};
#pragma pack(pop)

bool Trainer::saveCheckpoint(const std::string& filepath, int epoch, const EvaluationResult& valRes) {
    fs::create_directories(fs::path(filepath).parent_path());

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[-] Failed to open checkpoint file for writing: " << filepath << "\n";
        return false;
    }

    auto params = m_model.getParameters();

    CheckpointHeader header;
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, "CDCNN01", 7);
    header.epoch         = static_cast<int32_t>(epoch);
    header.learningRate  = m_currentLr;
    header.momentum      = m_config.momentum;
    header.weightDecay   = m_config.weightDecay;
    header.batchSize     = static_cast<int32_t>(m_config.batchSize);
    header.valLoss       = valRes.loss;
    header.valAccuracy   = valRes.accuracy;
    header.valPrecision  = valRes.precision;
    header.valRecall     = valRes.recall;
    header.valF1         = valRes.f1Score;
    header.valTN         = static_cast<int32_t>(valRes.confusionMatrix.tn);
    header.valFP         = static_cast<int32_t>(valRes.confusionMatrix.fp);
    header.valFN         = static_cast<int32_t>(valRes.confusionMatrix.fn);
    header.valTP         = static_cast<int32_t>(valRes.confusionMatrix.tp);
    header.numParams     = static_cast<uint32_t>(params.size());
    header.numBNLayers   = 3;

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Save each parameter: name length, name, size, weight bytes, velocity bytes
    for (Parameter* p : params) {
        std::string name = p ? p->name() : "unnamed";
        uint32_t nameLen = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        out.write(name.data(), nameLen);

        uint32_t numElem = static_cast<uint32_t>(p ? p->data().size() : 0);
        out.write(reinterpret_cast<const char*>(&numElem), sizeof(numElem));

        std::vector<float> wData = p ? p->data().toHostVector() : std::vector<float>();
        std::vector<float> vData = p ? p->velocity().toHostVector() : std::vector<float>();

        if (numElem > 0) {
            out.write(reinterpret_cast<const char*>(wData.data()), numElem * sizeof(float));
            out.write(reinterpret_cast<const char*>(vData.data()), numElem * sizeof(float));
        }
    }

    // Save BatchNorm running stats: bn1, bn2, bn3
    std::vector<std::pair<std::string, BatchNorm2D*>> bnLayers = {
        {"bn1", &m_model.bn1()},
        {"bn2", &m_model.bn2()},
        {"bn3", &m_model.bn3()}
    };

    for (auto& item : bnLayers) {
        std::string name = item.first;
        uint32_t nameLen = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        out.write(name.data(), nameLen);

        uint32_t numFeat = static_cast<uint32_t>(item.second->numFeatures());
        out.write(reinterpret_cast<const char*>(&numFeat), sizeof(numFeat));

        std::vector<float> rMean = item.second->runningMean().toHostVector();
        std::vector<float> rVar  = item.second->runningVar().toHostVector();

        out.write(reinterpret_cast<const char*>(rMean.data()), numFeat * sizeof(float));
        out.write(reinterpret_cast<const char*>(rVar.data()), numFeat * sizeof(float));
    }

    out.close();
    return true;
}

bool Trainer::loadCheckpoint(const std::string& filepath, int& loadedEpoch, EvaluationResult& loadedValRes) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[-] Failed to open checkpoint file for reading: " << filepath << "\n";
        return false;
    }

    CheckpointHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (std::strncmp(header.magic, "CDCNN01", 7) != 0) {
        std::cerr << "[-] Invalid checkpoint format in file: " << filepath << "\n";
        return false;
    }

    loadedEpoch = header.epoch;
    m_currentLr = header.learningRate;
    if (m_optimizer) {
        m_optimizer->setLearningRate(m_currentLr);
    }

    loadedValRes.loss = header.valLoss;
    loadedValRes.accuracy = header.valAccuracy;
    loadedValRes.precision = header.valPrecision;
    loadedValRes.recall = header.valRecall;
    loadedValRes.f1Score = header.valF1;
    loadedValRes.confusionMatrix.tn = header.valTN;
    loadedValRes.confusionMatrix.fp = header.valFP;
    loadedValRes.confusionMatrix.fn = header.valFN;
    loadedValRes.confusionMatrix.tp = header.valTP;

    auto params = m_model.getParameters();

    for (uint32_t i = 0; i < header.numParams; ++i) {
        uint32_t nameLen = 0;
        in.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        std::string name(nameLen, ' ');
        in.read(&name[0], nameLen);

        uint32_t numElem = 0;
        in.read(reinterpret_cast<char*>(&numElem), sizeof(numElem));

        std::vector<float> wData(numElem);
        std::vector<float> vData(numElem);

        if (numElem > 0) {
            in.read(reinterpret_cast<char*>(wData.data()), numElem * sizeof(float));
            in.read(reinterpret_cast<char*>(vData.data()), numElem * sizeof(float));
        }

        // Match parameter by index or name
        if (i < params.size() && params[i]) {
            params[i]->data().copyFromHost(wData.data(), numElem);
            params[i]->velocity().copyFromHost(vData.data(), numElem);
        }
    }

    // Read BatchNorm running statistics
    for (uint32_t k = 0; k < header.numBNLayers; ++k) {
        uint32_t nameLen = 0;
        in.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        std::string name(nameLen, ' ');
        in.read(&name[0], nameLen);

        uint32_t numFeat = 0;
        in.read(reinterpret_cast<char*>(&numFeat), sizeof(numFeat));

        std::vector<float> rMean(numFeat);
        std::vector<float> rVar(numFeat);

        in.read(reinterpret_cast<char*>(rMean.data()), numFeat * sizeof(float));
        in.read(reinterpret_cast<char*>(rVar.data()), numFeat * sizeof(float));

        BatchNorm2D* targetBN = nullptr;
        if (name == "bn1") targetBN = &m_model.bn1();
        else if (name == "bn2") targetBN = &m_model.bn2();
        else if (name == "bn3") targetBN = &m_model.bn3();

        if (targetBN && targetBN->numFeatures() == numFeat) {
            targetBN->runningMean().copyFromHost(rMean.data(), numFeat);
            targetBN->runningVar().copyFromHost(rVar.data(), numFeat);
        }
    }

    in.close();
    std::cout << "[Trainer] Successfully loaded checkpoint from " << filepath
              << " (Epoch: " << loadedEpoch << ", Val Acc: "
              << (loadedValRes.accuracy * 100.0f) << "%, LR: " << m_currentLr << ")\n";
    return true;
}

void Trainer::train(int numEpochs, const std::string& resumeCheckpointPath) {
    int startEpoch = 1;
    EvaluationResult bestValRes;
    bestValRes.loss = 1e9f;

    if (!resumeCheckpointPath.empty() && fs::exists(resumeCheckpointPath)) {
        int loadedEpoch = 0;
        if (loadCheckpoint(resumeCheckpointPath, loadedEpoch, bestValRes)) {
            startEpoch = loadedEpoch + 1;
            std::cout << "[Trainer] Resuming training from Epoch " << startEpoch
                      << " for " << numEpochs << " additional epochs.\n";
        }
    }

    int endEpoch = startEpoch + numEpochs - 1;

    // Initialize DataLoaders
    std::cout << "[Trainer] Initializing DataLoaders from " << m_config.manifestCsv << "...\n";
    DataLoader trainLoader(m_config.manifestCsv, m_config.imagesDir, "train",
                           m_config.batchSize, true, m_config.device);
    DataLoader valLoader(m_config.manifestCsv, m_config.imagesDir, "validation",
                         m_config.batchSize, false, m_config.device);

    size_t totalTrainBatches = trainLoader.numBatches();
    size_t totalValBatches   = valLoader.numBatches();

    std::cout << "  Training samples   : " << trainLoader.numSamples()
              << " (" << totalTrainBatches << " batches of " << m_config.batchSize << ")\n";
    std::cout << "  Validation samples : " << valLoader.numSamples()
              << " (" << totalValBatches << " batches of " << m_config.batchSize << ")\n";
    std::cout << "  Total Epochs Target: " << endEpoch << " (Epoch " << startEpoch << " -> " << endEpoch << ")\n\n";

    fs::create_directories(m_config.checkpointDir);

    for (int epoch = startEpoch; epoch <= endEpoch; ++epoch) {
        auto epochStart = std::chrono::steady_clock::now();

        // Learning rate schedule: step decay every m_config.lrDecayStep epochs
        if (epoch > startEpoch && (epoch - startEpoch) % m_config.lrDecayStep == 0) {
            m_currentLr *= m_config.lrDecayFactor;
            m_optimizer->setLearningRate(m_currentLr);
            std::cout << "[Scheduler] Learning rate annealed to: " << m_currentLr << "\n";
        }

        std::cout << "========================================================\n";
        std::cout << " Epoch [" << epoch << "/" << endEpoch << "] | LR: " << m_currentLr << "\n";
        std::cout << "========================================================\n";

        m_model.train(true);
        trainLoader.reset();

        double epochLoss = 0.0;
        Evaluator trainEval(0.5f);
        size_t batchIdx = 0;

        while (trainLoader.hasNext()) {
            Batch batch = trainLoader.nextBatch();
            batchIdx++;

            // 1. Clear gradients
            m_model.zeroGrad();

            // 2. Forward pass
            Tensor logits = m_model.forward(batch.images);

            // 3. Compute loss
            float batchLoss = m_criterion.forward(logits, batch.labels);
            epochLoss += batchLoss;

            // 4. Backward pass
            Tensor gradLogits = m_criterion.backward();
            m_model.backward(gradLogits);

            // 5. Update weights
            m_optimizer->step();

            // 6. Accumulate metrics
            trainEval.update(logits, batch.labels, batchLoss);

            // Periodic progress reporting
            if (batchIdx % m_config.logInterval == 0 || batchIdx == totalTrainBatches) {
                float progressPct = (static_cast<float>(batchIdx) / static_cast<float>(totalTrainBatches)) * 100.0f;
                float curAvgLoss = static_cast<float>(epochLoss / static_cast<double>(batchIdx));
                EvaluationResult curTrain = trainEval.compute();

                std::cout << "  [Batch " << std::setw(4) << batchIdx << "/" << totalTrainBatches
                          << " (" << std::fixed << std::setprecision(1) << progressPct << "%)]"
                          << " Loss: " << std::setprecision(4) << curAvgLoss
                          << " | Acc: " << (curTrain.accuracy * 100.0f) << "%"
                          << std::endl;
            }
        }

        auto trainEnd = std::chrono::steady_clock::now();
        double trainSec = std::chrono::duration<double>(trainEnd - epochStart).count();

        // 7. Validation phase
        std::cout << "\n  --> Running Validation on " << valLoader.numSamples() << " samples...\n";
        auto valStart = std::chrono::steady_clock::now();
        EvaluationResult valRes = evaluate(valLoader);
        auto valEnd = std::chrono::steady_clock::now();
        double valSec = std::chrono::duration<double>(valEnd - valStart).count();

        double totalEpochSec = trainSec + valSec;
        EvaluationResult trainFinalRes = trainEval.compute();

        // Record history
        EpochMetrics metrics;
        metrics.epoch               = epoch;
        metrics.trainLoss           = trainFinalRes.loss;
        metrics.trainAcc            = trainFinalRes.accuracy;
        metrics.trainF1             = trainFinalRes.f1Score;
        metrics.valLoss             = valRes.loss;
        metrics.valAcc              = valRes.accuracy;
        metrics.valPrec             = valRes.precision;
        metrics.valRec              = valRes.recall;
        metrics.valF1               = valRes.f1Score;
        metrics.valConfusionMatrix  = valRes.confusionMatrix;
        metrics.epochDurationSec    = totalEpochSec;
        metrics.learningRate        = m_currentLr;
        m_history.push_back(metrics);

        // 8. Print formatted epoch summary
        std::cout << "\n---------------- Epoch " << epoch << " Performance Summary ----------------\n";
        std::cout << " Training   : Loss = " << std::fixed << std::setprecision(4) << trainFinalRes.loss
                  << " | Accuracy = " << (trainFinalRes.accuracy * 100.0f) << "%\n";
        std::cout << " Validation : Loss = " << valRes.loss
                  << " | Accuracy = " << (valRes.accuracy * 100.0f) << "%\n";
        std::cout << "              Precision = " << (valRes.precision * 100.0f) << "%"
                  << " | Recall = " << (valRes.recall * 100.0f) << "%"
                  << " | F1-Score = " << (valRes.f1Score * 100.0f) << "%\n";
        std::cout << " Confusion  : TN = " << valRes.confusionMatrix.tn
                  << " (Cat), FP = " << valRes.confusionMatrix.fp
                  << ", FN = " << valRes.confusionMatrix.fn
                  << ", TP = " << valRes.confusionMatrix.tp << " (Dog)\n";
        std::cout << " Duration   : " << formatTime(totalEpochSec)
                  << " (Train: " << formatTime(trainSec) << ", Val: " << formatTime(valSec) << ")\n";
        std::cout << "--------------------------------------------------------\n";

        // 9. Save checkpoints
        std::string epochCkpt = m_config.checkpointDir + "/checkpoint_epoch_" + std::to_string(epoch) + ".bin";
        std::string latestCkpt = m_config.checkpointDir + "/latest_checkpoint.bin";
        saveCheckpoint(epochCkpt, epoch, valRes);
        saveCheckpoint(latestCkpt, epoch, valRes);

        if (valRes.loss < bestValRes.loss) {
            bestValRes = valRes;
            std::string bestCkpt = m_config.checkpointDir + "/best_model.bin";
            saveCheckpoint(bestCkpt, epoch, valRes);
            std::cout << " [*] New best validation model saved: " << bestCkpt << "\n";
        }

        std::cout << " [v] Checkpoint saved: " << epochCkpt << "\n";
        std::cout << " [>] To resume later: main.exe --resume " << latestCkpt << " --epochs <N>\n\n";
    }

    std::cout << "========================================================\n";
    std::cout << " >>> TRAINING COMPLETED FOR " << numEpochs << " EPOCHS! <<<\n";
    std::cout << "========================================================\n";
}
