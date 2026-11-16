#include "training/Trainer.hpp"
#include "core/CUDABackend.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static std::string resolvePath(const std::string& candidateA, const std::string& candidateB) {
    if (fs::exists(candidateA)) return candidateA;
    if (fs::exists(candidateB)) return candidateB;
    return candidateA;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "       CatDogCNN: C++ & CUDA Deep Learning Framework    \n";
    std::cout << "========================================================\n";

    bool isTrainMode = true;
    bool isSmokeTestOnly = false;
    int epochs = 5;
    int batchSize = 8;
    float lr = 0.01f;
    std::string resumePath = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--smoke-test") {
            isSmokeTestOnly = true;
            isTrainMode = false;
        } else if (arg == "--train") {
            isTrainMode = true;
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batchSize = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--resume" && i + 1 < argc) {
            resumePath = argv[++i];
        }
    }

    // Configure Trainer
    TrainingConfig config;
    config.batchSize = batchSize;
    config.learningRate = lr;
    config.momentum = 0.9f;
    config.weightDecay = 1e-4f;
    config.epochs = epochs;
    config.imagesDir = resolvePath("dataset/processed/images", "../dataset/processed/images");
    config.manifestCsv = resolvePath("dataset/split_manifest.csv", "../dataset/split_manifest.csv");
    config.checkpointDir = resolvePath("checkpoints", "../checkpoints");
    config.device = DeviceType::CUDA;

    std::cout << "Configuration Summary:\n";
    std::cout << "  Images Directory   : " << config.imagesDir << "\n";
    std::cout << "  Split Manifest     : " << config.manifestCsv << "\n";
    std::cout << "  Checkpoint Storage : " << config.checkpointDir << "\n";
    std::cout << "  Batch Size         : " << config.batchSize << "\n";
    std::cout << "  Initial LR         : " << config.learningRate << "\n";
    std::cout << "  Momentum           : " << config.momentum << "\n";
    std::cout << "  Weight Decay       : " << config.weightDecay << "\n";
    std::cout << "  Epochs Target      : " << config.epochs << "\n";
    if (!resumePath.empty()) {
        std::cout << "  Resume Checkpoint  : " << resumePath << "\n";
    }
    std::cout << "\n";

    Trainer trainer(config);

    // 1. Run smoke test first
    bool smokePass = trainer.runSmokeTest();
    if (!smokePass) {
        std::cerr << "[-] Pre-training smoke test failed! Aborting.\n";
        return 1;
    }

    if (isSmokeTestOnly) {
        std::cout << "[+] Smoke test completed successfully. Exiting as requested.\n";
        return 0;
    }

    // 2. Execute actual training
    std::cout << "\nStarting training pipeline for " << epochs << " epochs on NVIDIA GPU...\n\n";
    trainer.train(epochs, resumePath);

    return 0;
}
