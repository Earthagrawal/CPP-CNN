#pragma once

#include "Tensor.hpp"
#include <string>

class Parameter {
public:
    Parameter();
    explicit Parameter(const Tensor& data, bool requiresGrad = true, const std::string& name = "");

    Tensor& data() { return m_data; }
    const Tensor& data() const { return m_data; }

    Tensor& grad() { return m_grad; }
    const Tensor& grad() const { return m_grad; }

    Tensor& velocity() { return m_velocity; }
    const Tensor& velocity() const { return m_velocity; }

    bool requiresGrad() const { return m_requiresGrad; }
    void setRequiresGrad(bool req) { m_requiresGrad = req; }

    const std::string& name() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

    void zeroGrad();
    void toDevice(DeviceType dev);
    DeviceType device() const { return m_data.device(); }

private:
    Tensor m_data;
    Tensor m_grad;
    Tensor m_velocity;
    bool m_requiresGrad;
    std::string m_name;
};

