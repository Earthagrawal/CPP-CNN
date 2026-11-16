#include "Linear.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

Linear::Linear(size_t inFeatures, size_t outFeatures, bool useBias, DeviceType device, const std::string& name)
    : Layer(name),
      m_inFeatures(inFeatures),
      m_outFeatures(outFeatures),
      m_useBias(useBias),
      m_weight(Tensor({outFeatures, inFeatures}, device), true, name + ".weight"),
      m_bias(useBias ? Tensor({outFeatures}, device) : Tensor(), useBias, name + ".bias") {
    initializeWeights();
}

void Linear::initializeWeights() {
    // Kaiming / He uniform initialization: limit = sqrt(6 / inFeatures)
    float limit = std::sqrt(6.0f / static_cast<float>(m_inFeatures));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    size_t wSize = m_outFeatures * m_inFeatures;
    std::vector<float> h_weights(wSize);
    for (size_t i = 0; i < wSize; ++i) {
        h_weights[i] = dist(rng);
    }
    m_weight.data().copyFromHost(h_weights.data(), wSize);

    if (m_useBias) {
        m_bias.data().zero();
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (input.dim() < 2) {
        throw std::invalid_argument("Linear input must have at least 2 dimensions [Batch, InFeatures]");
    }
    size_t batchSize = input.shape()[0];
    size_t inFeats = input.size() / batchSize;
    if (inFeats != m_inFeatures) {
        throw std::invalid_argument("Input features (" + std::to_string(inFeats) + ") do not match layer inFeatures (" + std::to_string(m_inFeatures) + ")");
    }

    m_cachedInput = input;
    Tensor output({batchSize, m_outFeatures}, input.device());

    // Y = X * W^T
    // A = X [batchSize x inFeatures]
    // B = W [outFeatures x inFeatures]
    // C = Y [batchSize x outFeatures]
    CUDABackend::sgemm(false, true,
                       static_cast<int>(batchSize),
                       static_cast<int>(m_outFeatures),
                       static_cast<int>(m_inFeatures),
                       1.0f,
                       input.data(), static_cast<int>(m_inFeatures),
                       m_weight.data().data(), static_cast<int>(m_inFeatures),
                       0.0f,
                       output.data(), static_cast<int>(m_outFeatures),
                       input.device());

    if (m_useBias) {
        CUDABackend::addBias2D(output.data(), m_bias.data().data(),
                               static_cast<int>(batchSize),
                               static_cast<int>(m_outFeatures),
                               input.device());
    }

    return output;
}

Tensor Linear::backward(const Tensor& gradOutput) {
    size_t batchSize = gradOutput.shape()[0];
    Tensor gradInput(m_cachedInput.shape(), m_cachedInput.device());

    // 1. gradInput = gradOutput * W: [batchSize x outFeatures] * [outFeatures x inFeatures]
    CUDABackend::sgemm(false, false,
                       static_cast<int>(batchSize),
                       static_cast<int>(m_inFeatures),
                       static_cast<int>(m_outFeatures),
                       1.0f,
                       gradOutput.data(), static_cast<int>(m_outFeatures),
                       m_weight.data().data(), static_cast<int>(m_inFeatures),
                       0.0f,
                       gradInput.data(), static_cast<int>(m_inFeatures),
                       gradOutput.device());

    // 2. gradWeight += gradOutput^T * X: [outFeatures x batchSize] * [batchSize x inFeatures]
    if (m_weight.requiresGrad()) {
        CUDABackend::sgemm(true, false,
                           static_cast<int>(m_outFeatures),
                           static_cast<int>(m_inFeatures),
                           static_cast<int>(batchSize),
                           1.0f,
                           gradOutput.data(), static_cast<int>(m_outFeatures),
                           m_cachedInput.data(), static_cast<int>(m_inFeatures),
                           1.0f, // accumulate into grad
                           m_weight.grad().data(), static_cast<int>(m_inFeatures),
                           gradOutput.device());
    }

    // 3. gradBias += sum_b gradOutput[b, :]
    if (m_useBias && m_bias.requiresGrad()) {
        CUDABackend::sumRows(gradOutput.data(), m_bias.grad().data(),
                             static_cast<int>(batchSize),
                             static_cast<int>(m_outFeatures),
                             gradOutput.device());
    }

    return gradInput;
}

std::vector<Parameter*> Linear::getParameters() {
    std::vector<Parameter*> params = { &m_weight };
    if (m_useBias) {
        params.push_back(&m_bias);
    }
    return params;
}

void Linear::toDevice(DeviceType dev) {
    m_weight.toDevice(dev);
    if (m_useBias) {
        m_bias.toDevice(dev);
    }
    m_cachedInput.toDevice(dev);
}
