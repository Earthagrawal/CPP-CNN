#pragma once

#include "Layer.hpp"
#include "../core/Parameter.hpp"

class BatchNorm2D : public Layer {
public:
    BatchNorm2D(size_t numFeatures, float eps = 1e-5f, float momentum = 0.1f,
                DeviceType device = DeviceType::CPU,
                const std::string& name = "BatchNorm2D");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Parameter*> getParameters() override;
    void toDevice(DeviceType dev) override;

    Parameter& gamma() { return m_gamma; }
    const Parameter& gamma() const { return m_gamma; }

    Parameter& beta() { return m_beta; }
    const Parameter& beta() const { return m_beta; }

    Tensor& runningMean() { return m_runningMean; }
    const Tensor& runningMean() const { return m_runningMean; }

    Tensor& runningVar() { return m_runningVar; }
    const Tensor& runningVar() const { return m_runningVar; }

    size_t numFeatures() const { return m_numFeatures; }
    float eps() const { return m_eps; }
    float momentum() const { return m_momentum; }

private:
    size_t m_numFeatures;
    float m_eps;
    float m_momentum;

    Parameter m_gamma;
    Parameter m_beta;

    Tensor m_runningMean;
    Tensor m_runningVar;

    // Cache for backward pass
    Tensor m_cachedInput;
    Tensor m_saveMean;
    Tensor m_saveInvVar;
};
