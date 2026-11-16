#pragma once

#include "Layer.hpp"
#include "../core/Parameter.hpp"

class Conv2D : public Layer {
public:
    Conv2D(size_t inChannels, size_t outChannels,
           size_t kernelSize, size_t stride = 1, size_t padding = 0,
           bool useBias = true, DeviceType device = DeviceType::CPU,
           const std::string& name = "Conv2D");

    Conv2D(size_t inChannels, size_t outChannels,
           size_t kernelH, size_t kernelW,
           size_t strideH, size_t strideW,
           size_t padH, size_t padW,
           bool useBias = true, DeviceType device = DeviceType::CPU,
           const std::string& name = "Conv2D");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Parameter*> getParameters() override;
    void toDevice(DeviceType dev) override;

    Parameter& weight() { return m_weight; }
    Parameter& bias() { return m_bias; }
    size_t inChannels() const { return m_inChannels; }
    size_t outChannels() const { return m_outChannels; }

private:
    size_t m_inChannels;
    size_t m_outChannels;
    size_t m_kH;
    size_t m_kW;
    size_t m_strideH;
    size_t m_strideW;
    size_t m_padH;
    size_t m_padW;
    bool m_useBias;

    Parameter m_weight; // [outChannels, inChannels, kH, kW]
    Parameter m_bias;   // [outChannels]
    Tensor m_cachedInput;

    void initializeWeights();
};

