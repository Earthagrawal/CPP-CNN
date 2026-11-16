#include "Parameter.hpp"

Parameter::Parameter()
    : m_data(), m_grad(), m_velocity(), m_requiresGrad(true), m_name("") {}

Parameter::Parameter(const Tensor& data, bool requiresGrad, const std::string& name)
    : m_data(data),
      m_grad(data.shape(), 0.0f, data.device()),
      m_velocity(data.shape(), 0.0f, data.device()),
      m_requiresGrad(requiresGrad),
      m_name(name) {}

void Parameter::zeroGrad() {
    if (m_requiresGrad && !m_grad.empty()) {
        m_grad.zero();
    }
}

void Parameter::toDevice(DeviceType dev) {
    m_data.toDevice(dev);
    m_grad.toDevice(dev);
    m_velocity.toDevice(dev);
}

