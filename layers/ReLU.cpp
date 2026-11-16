#include "ReLU.hpp"

ReLU::ReLU(const std::string& name) : Layer(name) {}

Tensor ReLU::forward(const Tensor& input) {
    m_cachedInput = input;
    Tensor output(input.shape(), input.device());
    CUDABackend::reluForward(input.data(), output.data(), input.size(), input.device());
    return output;
}

Tensor ReLU::backward(const Tensor& gradOutput) {
    Tensor gradInput(m_cachedInput.shape(), m_cachedInput.device());
    CUDABackend::reluBackward(m_cachedInput.data(), gradOutput.data(), gradInput.data(), gradOutput.size(), gradOutput.device());
    return gradInput;
}

void ReLU::toDevice(DeviceType dev) {
    m_cachedInput.toDevice(dev);
}

