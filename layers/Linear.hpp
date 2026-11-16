#pragma once

#include "Layer.hpp"
#include "../core/Parameter.hpp"

class Linear : public Layer {
public:
    Linear(size_t inFeatures, size_t outFeatures, bool useBias = true, DeviceType device = DeviceType::CPU, const std::string& name = "Linear");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Parameter*> getParameters() override;
    void toDevice(DeviceType dev) override;

    Parameter& weight() { return m_weight; }
    Parameter& bias() { return m_bias; }
    size_t inFeatures() const { return m_inFeatures; }
    size_t outFeatures() const { return m_outFeatures; }

private:
    size_t m_inFeatures;
    size_t m_outFeatures;
    bool m_useBias;
    Parameter m_weight; // [outFeatures, inFeatures]
    Parameter m_bias;   // [outFeatures]
    Tensor m_cachedInput;

    void initializeWeights();
};

