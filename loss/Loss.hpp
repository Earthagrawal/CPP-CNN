#pragma once

#include "../core/Tensor.hpp"
#include <string>

class Loss {
public:
    explicit Loss(const std::string& name = "") : m_name(name) {}
    virtual ~Loss() = default;

    virtual float forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor backward() = 0;

    const std::string& name() const { return m_name; }

protected:
    std::string m_name;
};

