#pragma once

#include "Optimizer.hpp"

class SGD : public Optimizer {
public:
    SGD(std::vector<Parameter*> parameters,
        float learningRate = 0.01f,
        float momentum = 0.0f,
        float weightDecay = 0.0f);

    void step() override;

    float momentum() const { return m_momentum; }
    void setMomentum(float momentum) { m_momentum = momentum; }

    float weightDecay() const { return m_weightDecay; }
    void setWeightDecay(float wd) { m_weightDecay = wd; }

private:
    float m_momentum;
    float m_weightDecay;
};

