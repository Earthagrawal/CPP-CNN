#pragma once

#include "../core/Tensor.hpp"
#include <cstddef>
#include <string>

struct ConfusionMatrix {
    int tn = 0; // True Negatives (Cat correctly predicted as Cat, label 0)
    int fp = 0; // False Positives (Cat incorrectly predicted as Dog, label 1)
    int fn = 0; // False Negatives (Dog incorrectly predicted as Cat, label 0)
    int tp = 0; // True Positives (Dog correctly predicted as Dog, label 1)

    int total() const { return tn + fp + fn + tp; }
};

struct EvaluationResult {
    float loss = 0.0f;
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1Score = 0.0f;
    ConfusionMatrix confusionMatrix;

    std::string summary() const;
};

class Evaluator {
public:
    explicit Evaluator(float threshold = 0.5f);

    void reset();

    // Accumulate batch predictions and ground truth
    void update(const Tensor& logits, const Tensor& targets, float batchLoss = -1.0f);

    // Compute aggregated metrics
    EvaluationResult compute() const;

    // Direct one-shot batch/dataset evaluation
    static EvaluationResult evaluate(const Tensor& logits, const Tensor& targets, float threshold = 0.5f);

private:
    float m_threshold;
    ConfusionMatrix m_cm;
    double m_totalLoss;
    size_t m_sampleCount;
};
