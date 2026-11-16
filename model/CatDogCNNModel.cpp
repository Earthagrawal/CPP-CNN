#include "CatDogCNNModel.hpp"
#include <stdexcept>

CatDogCNNModel::CatDogCNNModel(DeviceType device)
    : m_device(device),
      m_isTraining(true),
      // Block 1: Conv(3 -> 32) -> BN(32) -> ReLU -> MaxPool(2x2)
      m_conv1(3, 32, 3, 1, 1, true, device, "conv1"),
      m_bn1(32, 1e-5f, 0.1f, device, "bn1"),
      m_relu1("relu1"),
      m_pool1(2, 2, "pool1"),
      // Block 2: Conv(32 -> 64) -> BN(64) -> ReLU -> MaxPool(2x2)
      m_conv2(32, 64, 3, 1, 1, true, device, "conv2"),
      m_bn2(64, 1e-5f, 0.1f, device, "bn2"),
      m_relu2("relu2"),
      m_pool2(2, 2, "pool2"),
      // Block 3: Conv(64 -> 128) -> BN(128) -> ReLU -> MaxPool(2x2)
      m_conv3(64, 128, 3, 1, 1, true, device, "conv3"),
      m_bn3(128, 1e-5f, 0.1f, device, "bn3"),
      m_relu3("relu3"),
      m_pool3(2, 2, "pool3"),
      // Head: GAP -> Linear(128 -> 32) -> ReLU -> Linear(32 -> 1)
      m_gap("gap"),
      m_fc1(128, 32, true, device, "fc1"),
      m_relu4("relu4"),
      m_fc2(32, 1, true, device, "fc2") {
}

Tensor CatDogCNNModel::forward(const Tensor& input) {
    if (input.dim() != 4) {
        throw std::invalid_argument("CatDogCNNModel forward expects 4D input [N, 3, 224, 224]");
    }

    // Block 1
    Tensor x = m_conv1.forward(input);
    x = m_bn1.forward(x);
    x = m_relu1.forward(x);
    x = m_pool1.forward(x);

    // Block 2
    x = m_conv2.forward(x);
    x = m_bn2.forward(x);
    x = m_relu2.forward(x);
    x = m_pool2.forward(x);

    // Block 3
    x = m_conv3.forward(x);
    x = m_bn3.forward(x);
    x = m_relu3.forward(x);
    x = m_pool3.forward(x);

    // Head
    x = m_gap.forward(x);
    x = m_fc1.forward(x);
    x = m_relu4.forward(x);
    x = m_fc2.forward(x);

    return x; // raw logits [N, 1]
}

Tensor CatDogCNNModel::backward(const Tensor& gradOutput) {
    // Backward through Head
    Tensor grad = m_fc2.backward(gradOutput);
    grad = m_relu4.backward(grad);
    grad = m_fc1.backward(grad);
    grad = m_gap.backward(grad);

    // Backward through Block 3
    grad = m_pool3.backward(grad);
    grad = m_relu3.backward(grad);
    grad = m_bn3.backward(grad);
    grad = m_conv3.backward(grad);

    // Backward through Block 2
    grad = m_pool2.backward(grad);
    grad = m_relu2.backward(grad);
    grad = m_bn2.backward(grad);
    grad = m_conv2.backward(grad);

    // Backward through Block 1
    grad = m_pool1.backward(grad);
    grad = m_relu1.backward(grad);
    grad = m_bn1.backward(grad);
    grad = m_conv1.backward(grad);

    return grad;
}

std::vector<Parameter*> CatDogCNNModel::getParameters() {
    std::vector<Parameter*> params;
    auto addParams = [&params](const std::vector<Parameter*>& pList) {
        params.insert(params.end(), pList.begin(), pList.end());
    };

    addParams(m_conv1.getParameters());
    addParams(m_bn1.getParameters());
    addParams(m_conv2.getParameters());
    addParams(m_bn2.getParameters());
    addParams(m_conv3.getParameters());
    addParams(m_bn3.getParameters());
    addParams(m_fc1.getParameters());
    addParams(m_fc2.getParameters());

    return params;
}

void CatDogCNNModel::zeroGrad() {
    for (auto* p : getParameters()) {
        if (p) {
            p->zeroGrad();
        }
    }
}

void CatDogCNNModel::toDevice(DeviceType dev) {
    m_device = dev;
    m_conv1.toDevice(dev);
    m_bn1.toDevice(dev);
    m_relu1.toDevice(dev);
    m_pool1.toDevice(dev);

    m_conv2.toDevice(dev);
    m_bn2.toDevice(dev);
    m_relu2.toDevice(dev);
    m_pool2.toDevice(dev);

    m_conv3.toDevice(dev);
    m_bn3.toDevice(dev);
    m_relu3.toDevice(dev);
    m_pool3.toDevice(dev);

    m_gap.toDevice(dev);
    m_fc1.toDevice(dev);
    m_relu4.toDevice(dev);
    m_fc2.toDevice(dev);
}

void CatDogCNNModel::train(bool mode) {
    m_isTraining = mode;
    m_conv1.train(mode);
    m_bn1.train(mode);
    m_relu1.train(mode);
    m_pool1.train(mode);

    m_conv2.train(mode);
    m_bn2.train(mode);
    m_relu2.train(mode);
    m_pool2.train(mode);

    m_conv3.train(mode);
    m_bn3.train(mode);
    m_relu3.train(mode);
    m_pool3.train(mode);

    m_gap.train(mode);
    m_fc1.train(mode);
    m_relu4.train(mode);
    m_fc2.train(mode);
}

void CatDogCNNModel::eval() {
    train(false);
}
