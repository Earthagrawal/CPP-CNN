#pragma once

#include "../core/Tensor.hpp"
#include "../core/Parameter.hpp"
#include <vector>
#include <string>

class Layer {
public:
    explicit Layer(const std::string& name = "") : m_name(name), m_isTraining(true) {}
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;

    virtual std::vector<Parameter*> getParameters() { return {}; }
    virtual void toDevice(DeviceType dev) { (void)dev; }

    virtual void train(bool mode = true) { m_isTraining = mode; }
    virtual void eval() { m_isTraining = false; }
    bool isTraining() const { return m_isTraining; }

    const std::string& name() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

protected:
    std::string m_name;
    bool m_isTraining;
};

