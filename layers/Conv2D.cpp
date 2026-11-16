#include "Conv2D.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

Conv2D::Conv2D(size_t inChannels, size_t outChannels,
               size_t kernelSize, size_t stride, size_t padding,
               bool useBias, DeviceType device, const std::string& name)
    : Conv2D(inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, useBias, device, name) {}

Conv2D::Conv2D(size_t inChannels, size_t outChannels,
               size_t kernelH, size_t kernelW,
               size_t strideH, size_t strideW,
               size_t padH, size_t padW,
               bool useBias, DeviceType device, const std::string& name)
    : Layer(name),
      m_inChannels(inChannels),
      m_outChannels(outChannels),
      m_kH(kernelH),
      m_kW(kernelW),
      m_strideH(strideH),
      m_strideW(strideW),
      m_padH(padH),
      m_padW(padW),
      m_useBias(useBias),
      m_weight(Tensor({outChannels, inChannels, kernelH, kernelW}, device), true, name + ".weight"),
      m_bias(useBias ? Tensor({outChannels}, device) : Tensor(), useBias, name + ".bias") {
    initializeWeights();
}

void Conv2D::initializeWeights() {
    size_t fanIn = m_inChannels * m_kH * m_kW;
    float limit = std::sqrt(6.0f / static_cast<float>(fanIn));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    size_t totalWeights = m_outChannels * m_inChannels * m_kH * m_kW;
    std::vector<float> h_weights(totalWeights);
    for (size_t i = 0; i < totalWeights; ++i) {
        h_weights[i] = dist(rng);
    }
    m_weight.data().copyFromHost(h_weights.data(), totalWeights);

    if (m_useBias) {
        m_bias.data().zero();
    }
}

Tensor Conv2D::forward(const Tensor& input) {
    if (input.dim() != 4) {
        throw std::invalid_argument("Conv2D input must be 4D [Batch, Channels, Height, Width]");
    }
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    if (C != m_inChannels) {
        throw std::invalid_argument("Input channels (" + std::to_string(C) + ") does not match Conv2D inChannels (" + std::to_string(m_inChannels) + ")");
    }

    size_t outH = (H + 2 * m_padH - m_kH) / m_strideH + 1;
    size_t outW = (W + 2 * m_padW - m_kW) / m_strideW + 1;

    m_cachedInput = input;
    Tensor output({N, m_outChannels, outH, outW}, input.device());

    const float* biasPtr = m_useBias ? m_bias.data().data() : nullptr;

    CUDABackend::conv2dForward(input.data(), m_weight.data().data(), biasPtr, output.data(),
                               static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
                               static_cast<int>(m_outChannels),
                               static_cast<int>(m_kH), static_cast<int>(m_kW),
                               static_cast<int>(m_strideH), static_cast<int>(m_strideW),
                               static_cast<int>(m_padH), static_cast<int>(m_padW),
                               input.device());

    return output;
}

Tensor Conv2D::backward(const Tensor& gradOutput) {
    size_t N = m_cachedInput.shape()[0];
    size_t C = m_cachedInput.shape()[1];
    size_t H = m_cachedInput.shape()[2];
    size_t W = m_cachedInput.shape()[3];

    Tensor gradInput(m_cachedInput.shape(), m_cachedInput.device());

    Tensor tempGradW;
    Tensor tempGradB;
    float* gwPtr = nullptr;
    float* gbPtr = nullptr;

    if (m_weight.requiresGrad()) {
        tempGradW = Tensor(m_weight.data().shape(), 0.0f, m_cachedInput.device());
        gwPtr = tempGradW.data();
    }
    if (m_useBias && m_bias.requiresGrad()) {
        tempGradB = Tensor(m_bias.data().shape(), 0.0f, m_cachedInput.device());
        gbPtr = tempGradB.data();
    }

    CUDABackend::conv2dBackward(m_cachedInput.data(), m_weight.data().data(), gradOutput.data(),
                                gradInput.data(), gwPtr, gbPtr,
                                static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
                                static_cast<int>(m_outChannels),
                                static_cast<int>(m_kH), static_cast<int>(m_kW),
                                static_cast<int>(m_strideH), static_cast<int>(m_strideW),
                                static_cast<int>(m_padH), static_cast<int>(m_padW),
                                m_cachedInput.device());

    if (m_weight.requiresGrad()) {
        CUDABackend::accumulate(m_weight.grad().data(), tempGradW.data(), m_weight.grad().size(), m_cachedInput.device());
    }

    if (m_useBias && m_bias.requiresGrad()) {
        CUDABackend::accumulate(m_bias.grad().data(), tempGradB.data(), m_bias.grad().size(), m_cachedInput.device());
    }

    return gradInput;
}

std::vector<Parameter*> Conv2D::getParameters() {
    std::vector<Parameter*> params = { &m_weight };
    if (m_useBias) {
        params.push_back(&m_bias);
    }
    return params;
}

void Conv2D::toDevice(DeviceType dev) {
    m_weight.toDevice(dev);
    if (m_useBias) {
        m_bias.toDevice(dev);
    }
    m_cachedInput.toDevice(dev);
}
