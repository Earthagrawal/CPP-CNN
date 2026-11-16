#pragma once

#include "../core/Parameter.hpp"
#include <vector>
#include <string>

class Optimizer {
public:
    explicit Optimizer(std::vector<Parameter*> parameters, float learningRate = 0.01f)
        : m_parameters(std::move(parameters)), m_learningRate(learningRate) {}

    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zeroGrad() {
        for (Parameter* p : m_parameters) {
            if (p) {
                p->zeroGrad();
            }
        }
    }

    float learningRate() const { return m_learningRate; }
    void setLearningRate(float lr) { m_learningRate = lr; }

protected:
    std::vector<Parameter*> m_parameters;
    float m_learningRate;
};

