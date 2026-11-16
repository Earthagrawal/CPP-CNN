#pragma once

#include "Loss.hpp"

class BCEWithLogitsLoss : public Loss {
public:
    explicit BCEWithLogitsLoss(const std::string& name = "BCEWithLogitsLoss");

    float forward(const Tensor& logits, const Tensor& targets) override;
    Tensor backward() override;

private:
    Tensor m_cachedLogits;
    Tensor m_cachedTargets;
};

