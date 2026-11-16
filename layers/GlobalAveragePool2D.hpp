#pragma once

#include "Layer.hpp"
#include <vector>

class GlobalAveragePool2D : public Layer {
public:
    explicit GlobalAveragePool2D(const std::string& name = "GlobalAveragePool2D");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    void toDevice(DeviceType dev) override;

private:
    std::vector<size_t> m_inputShape;
};

