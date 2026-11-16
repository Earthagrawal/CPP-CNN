#pragma once

#include "../model/CatDogCNNModel.hpp"
#include "../loss/BCEWithLogitsLoss.hpp"
#include "../optimizers/SGD.hpp"
#include "../data/DataLoader.hpp"
#include "../evaluation/Evaluator.hpp"
#include "../core/CUDABackend.hpp"

#include <string>
#include <vector>
#include <memory>
#include <chrono>

struct TrainingConfig {
    int batchSize           = 8;
    float learningRate      = 0.01f;
    float momentum          = 0.9f;
    float weightDecay       = 1e-4f;
    int epochs              = 5;
    std::string imagesDir   = "dataset/processed/images";
    std::string manifestCsv = "dataset/split_manifest.csv";
    std::string checkpointDir = "checkpoints";
    DeviceType device       = DeviceType::CUDA;
    int logInterval         = 250;     // Log progress every 250 batches
    float lrDecayFactor     = 0.5f;    // Learning rate annealing multiplier
    int lrDecayStep         = 2;       // Decay learning rate every 2 epochs
};

struct EpochMetrics {
    int epoch = 0;
    float trainLoss = 0.0f;
    float trainAcc = 0.0f;
    float trainF1 = 0.0f;
    float valLoss = 0.0f;
    float valAcc = 0.0f;
    float valPrec = 0.0f;
    float valRec = 0.0f;
    float valF1 = 0.0f;
    ConfusionMatrix valConfusionMatrix;
    double epochDurationSec = 0.0;
    float learningRate = 0.0f;
};

class Trainer {
public:
    explicit Trainer(const TrainingConfig& config = TrainingConfig());

    // Verification smoke test covering forward, backward, parameter updates,
    // BatchNorm running statistics updates and invariance during eval, and CUDA execution.
    bool runSmokeTest();

    // Run training for numEpochs. If resumeCheckpointPath is non-empty,
    // resumes state and trains for numEpochs starting from (loadedEpoch + 1).
    void train(int numEpochs = 5, const std::string& resumeCheckpointPath = "");

    // Run inference/evaluation over an entire DataLoader split in eval mode
    EvaluationResult evaluate(DataLoader& loader);

    // Save and load complete resumable checkpoints (weights, BN stats, velocities, config)
    bool saveCheckpoint(const std::string& filepath, int epoch, const EvaluationResult& valRes);
    bool loadCheckpoint(const std::string& filepath, int& loadedEpoch, EvaluationResult& loadedValRes);

    const TrainingConfig& config() const { return m_config; }
    const std::vector<EpochMetrics>& history() const { return m_history; }
    CatDogCNNModel& model() { return m_model; }

private:
    TrainingConfig m_config;
    CatDogCNNModel m_model;
    BCEWithLogitsLoss m_criterion;
    std::unique_ptr<SGD> m_optimizer;
    std::vector<EpochMetrics> m_history;
    float m_currentLr;

    void initializeDevice();
    void setupOptimizer();
    std::string formatTime(double seconds) const;
};
