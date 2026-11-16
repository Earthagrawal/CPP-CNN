#include "GlobalAveragePool2D.hpp"
#include <stdexcept>

GlobalAveragePool2D::GlobalAveragePool2D(const std::string& name)
    : Layer(name), m_inputShape({}) {}

Tensor GlobalAveragePool2D::forward(const Tensor& input) {
    if (input.dim() != 4) {
        throw std::invalid_argument("GlobalAveragePool2D requires 4D input [Batch, Channels, Height, Width]");
    }
    m_inputShape = input.shape();
    size_t N = m_inputShape[0];
    size_t C = m_inputShape[1];
    size_t H = m_inputShape[2];
    size_t W = m_inputShape[3];

    Tensor output({N, C}, input.device());

    CUDABackend::globalAvgPoolForward(input.data(), output.data(),
                                      static_cast<int>(N), static_cast<int>(C),
                                      static_cast<int>(H), static_cast<int>(W),
                                      input.device());

    return output;
}

Tensor GlobalAveragePool2D::backward(const Tensor& gradOutput) {
    if (m_inputShape.empty()) {
        throw std::runtime_error("GlobalAveragePool2D backward called before forward");
    }
    size_t N = m_inputShape[0];
    size_t C = m_inputShape[1];
    size_t H = m_inputShape[2];
    size_t W = m_inputShape[3];

    Tensor gradInput(m_inputShape, gradOutput.device());

    CUDABackend::globalAvgPoolBackward(gradOutput.data(), gradInput.data(),
                                       static_cast<int>(N), static_cast<int>(C),
                                       static_cast<int>(H), static_cast<int>(W),
                                       gradOutput.device());

    return gradInput;
}

void GlobalAveragePool2D::toDevice(DeviceType dev) {
    (void)dev;
}

