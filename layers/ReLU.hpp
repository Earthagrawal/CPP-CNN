#pragma once

#include "Layer.hpp"

class ReLU : public Layer {
public:
    explicit ReLU(const std::string& name = "ReLU");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    void toDevice(DeviceType dev) override;

private:
    Tensor m_cachedInput;
};

