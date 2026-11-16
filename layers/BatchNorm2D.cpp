#include "BatchNorm2D.hpp"
#include <stdexcept>

BatchNorm2D::BatchNorm2D(size_t numFeatures, float eps, float momentum,
                         DeviceType device, const std::string& name)
    : Layer(name),
      m_numFeatures(numFeatures),
      m_eps(eps),
      m_momentum(momentum),
      m_gamma(Tensor({numFeatures}, 1.0f, device), true, name + ".gamma"),
      m_beta(Tensor({numFeatures}, 0.0f, device), true, name + ".beta"),
      m_runningMean({numFeatures}, 0.0f, device),
      m_runningVar({numFeatures}, 1.0f, device),
      m_saveMean({numFeatures}, 0.0f, device),
      m_saveInvVar({numFeatures}, 0.0f, device) {
}

Tensor BatchNorm2D::forward(const Tensor& input) {
    if (input.dim() != 4) {
        throw std::invalid_argument("BatchNorm2D input must be 4D [Batch, Channels, Height, Width]");
    }
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    if (C != m_numFeatures) {
        throw std::invalid_argument("Input channels (" + std::to_string(C) +
                                   ") does not match BatchNorm2D numFeatures (" +
                                   std::to_string(m_numFeatures) + ")");
    }

    m_cachedInput = input;
    Tensor output(input.shape(), input.device());

    CUDABackend::batchNorm2dForward(
        input.data(),
        output.data(),
        m_gamma.data().data(),
        m_beta.data().data(),
        m_runningMean.data(),
        m_runningVar.data(),
        m_saveMean.data(),
        m_saveInvVar.data(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        m_eps,
        m_momentum,
        m_isTraining,
        input.device()
    );

    return output;
}

Tensor BatchNorm2D::backward(const Tensor& gradOutput) {
    if (gradOutput.shape() != m_cachedInput.shape()) {
        throw std::invalid_argument("BatchNorm2D backward: gradOutput shape mismatch with cached input");
    }

    size_t N = m_cachedInput.shape()[0];
    size_t C = m_cachedInput.shape()[1];
    size_t H = m_cachedInput.shape()[2];
    size_t W = m_cachedInput.shape()[3];

    Tensor gradInput(m_cachedInput.shape(), m_cachedInput.device());

    CUDABackend::batchNorm2dBackward(
        gradOutput.data(),
        m_cachedInput.data(),
        gradInput.data(),
        m_gamma.data().data(),
        m_gamma.requiresGrad() ? m_gamma.grad().data() : nullptr,
        m_beta.requiresGrad() ? m_beta.grad().data() : nullptr,
        m_saveMean.data(),
        m_saveInvVar.data(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        m_cachedInput.device()
    );

    return gradInput;
}

std::vector<Parameter*> BatchNorm2D::getParameters() {
    return { &m_gamma, &m_beta };
}

void BatchNorm2D::toDevice(DeviceType dev) {
    m_gamma.toDevice(dev);
    m_beta.toDevice(dev);
    m_runningMean.toDevice(dev);
    m_runningVar.toDevice(dev);
    m_saveMean.toDevice(dev);
    m_saveInvVar.toDevice(dev);
    m_cachedInput.toDevice(dev);
}
