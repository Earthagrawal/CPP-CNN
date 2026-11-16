#pragma once

#include "../layers/Conv2D.hpp"
#include "../layers/BatchNorm2D.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/MaxPool2D.hpp"
#include "../layers/GlobalAveragePool2D.hpp"
#include "../layers/Linear.hpp"
#include <vector>
#include <memory>

class CatDogCNNModel {
public:
    explicit CatDogCNNModel(DeviceType device = DeviceType::CPU);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput);

    std::vector<Parameter*> getParameters();
    void zeroGrad();
    void toDevice(DeviceType dev);

    void train(bool mode = true);
    void eval();
    bool isTraining() const { return m_isTraining; }
    DeviceType device() const { return m_device; }

    // Layer accessors
    Conv2D& conv1() { return m_conv1; }
    BatchNorm2D& bn1() { return m_bn1; }
    Conv2D& conv2() { return m_conv2; }
    BatchNorm2D& bn2() { return m_bn2; }
    Conv2D& conv3() { return m_conv3; }
    BatchNorm2D& bn3() { return m_bn3; }
    Linear& fc1() { return m_fc1; }
    Linear& fc2() { return m_fc2; }

private:
    DeviceType m_device;
    bool m_isTraining;

    // Block 1
    Conv2D m_conv1;
    BatchNorm2D m_bn1;
    ReLU m_relu1;
    MaxPool2D m_pool1;

    // Block 2
    Conv2D m_conv2;
    BatchNorm2D m_bn2;
    ReLU m_relu2;
    MaxPool2D m_pool2;

    // Block 3
    Conv2D m_conv3;
    BatchNorm2D m_bn3;
    ReLU m_relu3;
    MaxPool2D m_pool3;

    // Global Pooling & Classifier
    GlobalAveragePool2D m_gap;
    Linear m_fc1;
    ReLU m_relu4;
    Linear m_fc2;
};
