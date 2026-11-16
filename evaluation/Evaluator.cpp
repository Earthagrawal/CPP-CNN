#include "Evaluator.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

std::string EvaluationResult::summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "================ Evaluation Metrics ================\n";
    oss << " Average BCE Loss : " << loss << "\n";
    oss << " Accuracy         : " << (accuracy * 100.0f) << "%\n";
    oss << " Precision        : " << (precision * 100.0f) << "%\n";
    oss << " Recall           : " << (recall * 100.0f) << "%\n";
    oss << " F1-Score         : " << (f1Score * 100.0f) << "%\n";
    oss << "---------------- Confusion Matrix ------------------\n";
    oss << "                  Predicted Cat (0)   Predicted Dog (1)\n";
    oss << " Actual Cat (0)   TN: " << std::setw(6) << confusionMatrix.tn
        << "        FP: " << std::setw(6) << confusionMatrix.fp << "\n";
    oss << " Actual Dog (1)   FN: " << std::setw(6) << confusionMatrix.fn
        << "        TP: " << std::setw(6) << confusionMatrix.tp << "\n";
    oss << " Total Evaluated  : " << confusionMatrix.total() << " samples\n";
    oss << "====================================================";
    return oss.str();
}

Evaluator::Evaluator(float threshold)
    : m_threshold(threshold),
      m_totalLoss(0.0),
      m_sampleCount(0) {
    reset();
}

void Evaluator::reset() {
    m_cm = ConfusionMatrix();
    m_totalLoss = 0.0;
    m_sampleCount = 0;
}

void Evaluator::update(const Tensor& logits, const Tensor& targets, float batchLoss) {
    if (logits.size() != targets.size()) {
        throw std::invalid_argument("Evaluator::update: size mismatch between logits and targets");
    }

    std::vector<float> h_logits = logits.toHostVector();
    std::vector<float> h_targets = targets.toHostVector();
    size_t n = h_logits.size();

    double curBatchLoss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float z = h_logits[i];
        float y = h_targets[i];

        // Sigmoid probability: p = 1 / (1 + exp(-z))
        float p = 1.0f / (1.0f + std::exp(-z));
        int pred = (p >= m_threshold) ? 1 : 0;
        int actual = (y >= 0.5f) ? 1 : 0;

        if (actual == 1 && pred == 1) {
            m_cm.tp++;
        } else if (actual == 0 && pred == 1) {
            m_cm.fp++;
        } else if (actual == 1 && pred == 0) {
            m_cm.fn++;
        } else {
            m_cm.tn++;
        }

        if (batchLoss < 0.0f) {
            // Compute numerically stable BCE per element
            float maxVal = (z > 0.0f) ? z : 0.0f;
            float term = maxVal - z * y + std::log1p(std::exp(-std::abs(z)));
            curBatchLoss += static_cast<double>(term);
        }
    }

    if (batchLoss >= 0.0f) {
        m_totalLoss += static_cast<double>(batchLoss) * n;
    } else {
        m_totalLoss += curBatchLoss;
    }

    m_sampleCount += n;
}

EvaluationResult Evaluator::compute() const {
    EvaluationResult res;
    res.confusionMatrix = m_cm;

    int totalSamples = m_cm.total();
    if (totalSamples > 0) {
        res.accuracy = static_cast<float>(m_cm.tp + m_cm.tn) / static_cast<float>(totalSamples);
    } else {
        res.accuracy = 0.0f;
    }

    // Precision = TP / (TP + FP)
    int predPositives = m_cm.tp + m_cm.fp;
    if (predPositives > 0) {
        res.precision = static_cast<float>(m_cm.tp) / static_cast<float>(predPositives);
    } else {
        res.precision = 0.0f;
    }

    // Recall = TP / (TP + FN)
    int actualPositives = m_cm.tp + m_cm.fn;
    if (actualPositives > 0) {
        res.recall = static_cast<float>(m_cm.tp) / static_cast<float>(actualPositives);
    } else {
        res.recall = 0.0f;
    }

    // F1 = 2 * (P * R) / (P + R)
    float prSum = res.precision + res.recall;
    if (prSum > 1e-7f) {
        res.f1Score = 2.0f * (res.precision * res.recall) / prSum;
    } else {
        res.f1Score = 0.0f;
    }

    if (m_sampleCount > 0) {
        res.loss = static_cast<float>(m_totalLoss / static_cast<double>(m_sampleCount));
    } else {
        res.loss = 0.0f;
    }

    return res;
}

EvaluationResult Evaluator::evaluate(const Tensor& logits, const Tensor& targets, float threshold) {
    Evaluator ev(threshold);
    ev.update(logits, targets);
    return ev.compute();
}
