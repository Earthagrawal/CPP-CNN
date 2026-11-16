#include "BCEWithLogitsLoss.hpp"
#include <stdexcept>

BCEWithLogitsLoss::BCEWithLogitsLoss(const std::string& name) : Loss(name) {}

float BCEWithLogitsLoss::forward(const Tensor& logits, const Tensor& targets) {
    if (logits.size() != targets.size()) {
        throw std::invalid_argument("Logits and targets sizes must match in BCEWithLogitsLoss");
    }
    m_cachedLogits = logits;
    m_cachedTargets = targets;

    float lossVal = 0.0f;
    CUDABackend::bceWithLogitsForward(logits.data(), targets.data(), &lossVal, logits.size(), logits.device());
    return lossVal;
}

Tensor BCEWithLogitsLoss::backward() {
    if (m_cachedLogits.empty() || m_cachedTargets.empty()) {
        throw std::runtime_error("BCEWithLogitsLoss backward called before forward");
    }
    Tensor gradLogits(m_cachedLogits.shape(), m_cachedLogits.device());
    CUDABackend::bceWithLogitsBackward(m_cachedLogits.data(), m_cachedTargets.data(), gradLogits.data(),
                                       m_cachedLogits.size(), m_cachedLogits.device());
    return gradLogits;
}

